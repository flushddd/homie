"""Phase-3 EE residual → arm joints via damped least-squares IK.

Homie ``legged_robot422`` path::

  pos_error = (target - ee) + delta_e3[:3]
  pos_error = clip(pos_error, ±0.1)
  q_arm ← DLS IK on clipped error; clamp to joint limits

Policy residual is still ``tanh * scale`` then clip. The important rate-limit
is the **Cartesian error clip** — without it IK chases the full grasp target
in one step and yanks the COM (unlike phase-1/2 smooth random-arm curriculum).

Inactive reach envs (``is_target=0``) hold arm joints at default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from homierl_mjlab.mdp.base_frame import (
  rotate_world_to_base,
  world_to_base,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_LEFT_ARM = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
)
_RIGHT_ARM = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
_LEFT_ARM_BODIES = (
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
)
_RIGHT_ARM_BODIES = (
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)
_ARM_AXES_LOCAL = (
  (0.0, 1.0, 0.0),
  (1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0),
  (0.0, 1.0, 0.0),
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, 0.0, 1.0),
)


def _quat_apply(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat_wxyz.unbind(-1)
  vx, vy, vz = vec.unbind(-1)
  tx = 2.0 * (y * vz - z * vy)
  ty = 2.0 * (z * vx - x * vz)
  tz = 2.0 * (x * vy - y * vx)
  return torch.stack(
    (
      vx + w * tx + y * tz - z * ty,
      vy + w * ty + z * tx - x * tz,
      vz + w * tz + x * ty - y * tx,
    ),
    dim=-1,
  )


class EeResidualIkAction(ActionTerm):
  """Homie-style clipped EE error → multi-iter DLS IK → joint targets."""

  cfg: EeResidualIkActionCfg

  def __init__(self, cfg: EeResidualIkActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self.robot = env.scene[cfg.entity_name]
    self._raw = torch.zeros(self.num_envs, 3, device=self.device)
    self._delta = torch.zeros(self.num_envs, 3, device=self.device)
    self._scale = torch.tensor(cfg.ee_delta_scale, device=self.device, dtype=torch.float32)
    self._clip = torch.tensor(cfg.ee_delta_clip, device=self.device, dtype=torch.float32)
    self._err_clip = torch.tensor(
      cfg.ee_error_clip, device=self.device, dtype=torch.float32
    )

    jnames = list(self.robot.joint_names)
    bnames = list(self.robot.body_names)

    def _jids(names: tuple[str, ...]) -> torch.Tensor:
      return torch.tensor(
        [jnames.index(n) for n in names], device=self.device, dtype=torch.long
      )

    def _bids(names: tuple[str, ...]) -> torch.Tensor:
      return torch.tensor(
        [bnames.index(n) for n in names], device=self.device, dtype=torch.long
      )

    self._left_ids = _jids(_LEFT_ARM)
    self._right_ids = _jids(_RIGHT_ARM)
    self._left_bodies = _bids(_LEFT_ARM_BODIES)
    self._right_bodies = _bids(_RIGHT_ARM_BODIES)
    self._left_ee = bnames.index("left_wrist_yaw_link")
    self._right_ee = bnames.index("right_wrist_yaw_link")
    self._axes_local = torch.tensor(
      _ARM_AXES_LOCAL, device=self.device, dtype=torch.float32
    )
    self._damping = float(cfg.ik_damping)
    self._max_dq = float(cfg.max_joint_delta)
    self._ik_iters = int(cfg.ik_iterations)
    self._null_k = float(cfg.nullspace_gain)
    # Soft joint limits for post-IK clamp (J, 2).
    limits = self.robot.data.soft_joint_pos_limits[0]
    self._left_lim = limits[self._left_ids].clone()
    self._right_lim = limits[self._right_ids].clone()
    # Smoothed joint command (like Homie arm goal hold / phase-1/2 interp).
    default = self.robot.data.default_joint_pos
    self._left_cmd = default[:, self._left_ids].clone()
    self._right_cmd = default[:, self._right_ids].clone()
    self._blend = float(cfg.joint_target_blend)
    self._use_policy_delta = bool(cfg.use_policy_delta)
    # Prefer upright/default wrist (idx 4–6) so position-only IK does not leave
    # the hand flopping; proximal joints get a lighter nullspace pull.
    n_arm = int(self._left_ids.numel())
    null_w = torch.ones(n_arm, device=self.device, dtype=torch.float32)
    if n_arm >= 7:
      null_w[4:7] = 2.5
    self._null_w = null_w

  @property
  def action_dim(self) -> int:
    return 3

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  @property
  def ee_delta(self) -> torch.Tensor:
    """Clipped residual in meters, Homie base frame."""
    return self._delta

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    if not self._use_policy_delta:
      # Play / debug: untrained residual only injects high-freq jitter.
      self._delta.zero_()
      return
    delta = self._scale * torch.tanh(self._raw)
    self._delta = torch.maximum(torch.minimum(delta, self._clip), -self._clip)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    default = self.robot.data.default_joint_pos
    if env_ids is None:
      self._raw.zero_()
      self._delta.zero_()
      self._left_cmd[:] = default[:, self._left_ids]
      self._right_cmd[:] = default[:, self._right_ids]
    else:
      self._raw[env_ids] = 0.0
      self._delta[env_ids] = 0.0
      self._left_cmd[env_ids] = default[env_ids][:, self._left_ids]
      self._right_cmd[env_ids] = default[env_ids][:, self._right_ids]

  def apply_actions(self) -> None:
    grasp = self._env.command_manager.get_term("grasp")
    arm = grasp.target_arm.squeeze(-1)
    # Reach mode only (is_target=1). Waist-only / loco keep arms at default.
    is_reach = grasp.is_target.squeeze(-1) > 0.5
    left_mask = is_reach & (arm > 0.5) & (arm < 1.5)
    right_mask = is_reach & (arm > 1.5)

    default = self.robot.data.default_joint_pos
    left_tgt = default[:, self._left_ids].clone()
    right_tgt = default[:, self._right_ids].clone()

    # Homie: error = (target - ee) + delta, then clip (rate-limit toward target).
    tgt_b = world_to_base(self.robot, grasp.target_pos_w)

    if bool(left_mask.any()):
      left_tgt = self._ik_toward_target(
        env_mask=left_mask,
        joint_ids=self._left_ids,
        body_ids=self._left_bodies,
        ee_body=self._left_ee,
        target_base=tgt_b,
        limits=self._left_lim,
        cmd_buf=self._left_cmd,
        out_targets=left_tgt,
      )
    if bool(right_mask.any()):
      right_tgt = self._ik_toward_target(
        env_mask=right_mask,
        joint_ids=self._right_ids,
        body_ids=self._right_bodies,
        ee_body=self._right_ee,
        target_base=tgt_b,
        limits=self._right_lim,
        cmd_buf=self._right_cmd,
        out_targets=right_tgt,
      )

    # Non-reach: ease commands back to default so play on/off is not a hard snap.
    idle = ~is_reach
    if bool(idle.any()):
      ids = idle.nonzero(as_tuple=False).flatten()
      a = self._blend
      self._left_cmd[ids] = (1.0 - a) * self._left_cmd[ids] + a * default[ids][
        :, self._left_ids
      ]
      self._right_cmd[ids] = (1.0 - a) * self._right_cmd[ids] + a * default[ids][
        :, self._right_ids
      ]
      left_tgt[ids] = self._left_cmd[ids]
      right_tgt[ids] = self._right_cmd[ids]

    self.robot.set_joint_position_target(left_tgt, joint_ids=self._left_ids)
    self.robot.set_joint_position_target(right_tgt, joint_ids=self._right_ids)

  def _ik_toward_target(
    self,
    *,
    env_mask: torch.Tensor,
    joint_ids: torch.Tensor,
    body_ids: torch.Tensor,
    ee_body: int,
    target_base: torch.Tensor,
    limits: torch.Tensor,
    cmd_buf: torch.Tensor,
    out_targets: torch.Tensor,
  ) -> torch.Tensor:
    ids = env_mask.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
      return out_targets

    # Always seed from measured q + measured J/ee (same pose). Seeding from
    # cmd_buf while J is at the real config makes the step inconsistent and
    # the arm can thrash.
    q = self.robot.data.joint_pos[ids][:, joint_ids].clone()
    q_pref = self.robot.data.default_joint_pos[ids][:, joint_ids]
    ee_w = self.robot.data.body_link_pos_w[ids, ee_body]
    ee_b = world_to_base(self.robot, ee_w, ids)
    # Homie: pos_error = (target - ee) + delta_e3, then clip ±0.1.
    err = (target_base[ids] - ee_b) + self._delta[ids]
    err = torch.maximum(torch.minimum(err, self._err_clip), -self._err_clip)

    m = int(ids.numel())
    n_arm = int(joint_ids.numel())
    eye3 = torch.eye(3, device=self.device, dtype=q.dtype).expand(m, 3, 3)
    eye_n = torch.eye(n_arm, device=self.device, dtype=q.dtype).expand(m, n_arm, n_arm)
    J = self._geometric_jacobian_base(ids, body_ids, ee_body)
    ee = ee_b
    null_w = self._null_w.to(dtype=q.dtype)
    for _ in range(self._ik_iters):
      Jt = J.transpose(1, 2)
      JJt = torch.bmm(J, Jt)
      A = JJt + (self._damping**2) * eye3
      # Recompute remaining error toward the same clipped step goal.
      step_goal = ee_b + err
      delta_x = torch.linalg.solve(A, (step_goal - ee).unsqueeze(-1)).squeeze(-1)
      dq = torch.bmm(Jt, delta_x.unsqueeze(-1)).squeeze(-1)
      # Nullspace → default pose (heavier on wrists) so the hand does not droop.
      if self._null_k > 0.0:
        # J# ≈ J^T (JJ^T + λ²I)^{-1}; N = I - J# J.
        A_inv_J = torch.linalg.solve(A, J)
        Jpinv = torch.bmm(Jt, A_inv_J)
        N = eye_n - Jpinv
        dq_null = self._null_k * null_w * (q_pref - q)
        dq = dq + torch.bmm(N, dq_null.unsqueeze(-1)).squeeze(-1)
      dq = torch.clamp(dq, -self._max_dq, self._max_dq)
      q = q + dq
      ee = ee + torch.bmm(J, dq.unsqueeze(-1)).squeeze(-1)

    # Homie: clamp to dof limits after IK.
    q = torch.maximum(torch.minimum(q, limits[:, 1]), limits[:, 0])

    # Blend toward IK solution (smooth vs hard snap).
    a = self._blend
    cmd_buf[ids] = (1.0 - a) * cmd_buf[ids] + a * q
    out_targets = out_targets.clone()
    out_targets[ids] = cmd_buf[ids]
    return out_targets

  def _geometric_jacobian_base(
    self,
    ids: torch.Tensor,
    body_ids: torch.Tensor,
    ee_body: int,
  ) -> torch.Tensor:
    """Positional Jacobian in Homie base frame ``(M, 3, n_arm)``."""
    ee_w = self.robot.data.body_link_pos_w[ids, ee_body]
    n_arm = int(body_ids.numel())
    J_w = torch.zeros(ids.numel(), 3, n_arm, device=self.device, dtype=ee_w.dtype)
    for i in range(n_arm):
      bi = int(body_ids[i].item())
      origin = self.robot.data.body_link_pos_w[ids, bi]
      quat = self.robot.data.body_link_quat_w[ids, bi]
      axis_l = self._axes_local[i].expand(ids.numel(), 3)
      axis_w = _quat_apply(quat, axis_l)
      J_w[:, :, i] = torch.cross(axis_w, ee_w - origin, dim=-1)
    J_b = torch.zeros_like(J_w)
    for i in range(n_arm):
      J_b[:, :, i] = rotate_world_to_base(self.robot, J_w[:, :, i], ids)
    return J_b


@dataclass(kw_only=True)
class EeResidualIkActionCfg(ActionTermCfg):
  """Config for Homie-style clipped EE error + DLS IK."""

  entity_name: str = "robot"
  # Policy residual (meters) after tanh.
  ee_delta_scale: tuple[float, float, float] = (0.03, 0.03, 0.05)
  ee_delta_clip: tuple[float, float, float] = (0.04, 0.04, 0.06)
  # Homie clips (target-ee+delta) to ±0.1 — this is the main yank prevention.
  ee_error_clip: tuple[float, float, float] = (0.10, 0.10, 0.10)
  ik_damping: float = 0.08
  max_joint_delta: float = 0.08
  ik_iterations: int = 4
  # 1.0 = apply IK q immediately; lower = interpolate (play / soft start).
  joint_target_blend: float = 0.35
  # Light pull on redundant DOFs (esp. wrists). Keep small — damped J# makes
  # N imperfect, and a large gain fights the position task (arm looks stuck).
  nullspace_gain: float = 0.1
  # If False, ignore policy EE residual (recommended for play on phase-2 ckpt).
  use_policy_delta: bool = True
  fd_eps: float = 1e-3

  def build(self, env: ManagerBasedRlEnv) -> EeResidualIkAction:
    return EeResidualIkAction(self, env)
