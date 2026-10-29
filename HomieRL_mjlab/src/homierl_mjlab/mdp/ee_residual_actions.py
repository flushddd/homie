"""Phase-3 EE pose delta → arm joints via multi-iteration DLS IK.

Policy outputs ``(Δx, Δy, Δz)`` in Homie **base** frame (pelvis + yaw):
a bounded residual relative to the anchored grasp target.

  raw → tanh → * scale → clamp to ±ee_delta_clip (meters)
  target_base = world_to_base(p_anchor)
  desired_base = target_base + delta
  q_arm ← DLS IK in base frame toward desired_base

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
  """3-D EE residual (base frame) → multi-iter geometric DLS IK."""

  cfg: EeResidualIkActionCfg

  def __init__(self, cfg: EeResidualIkActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self.robot = env.scene[cfg.entity_name]
    self._raw = torch.zeros(self.num_envs, 3, device=self.device)
    self._delta = torch.zeros(self.num_envs, 3, device=self.device)
    self._scale = torch.tensor(cfg.ee_delta_scale, device=self.device, dtype=torch.float32)
    self._clip = torch.tensor(cfg.ee_delta_clip, device=self.device, dtype=torch.float32)

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
    delta = self._scale * torch.tanh(self._raw)
    self._delta = torch.maximum(torch.minimum(delta, self._clip), -self._clip)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self._raw.zero_()
      self._delta.zero_()
    else:
      self._raw[env_ids] = 0.0
      self._delta[env_ids] = 0.0

  def apply_actions(self) -> None:
    grasp = self._env.command_manager.get_term("grasp")
    arm = grasp.target_arm.squeeze(-1)
    # Reach mode only (is_target=1). Waist-only / loco keep arms at default —
    # gating on is_waist yanked ~80% of envs toward far targets and caused falls.
    is_reach = grasp.is_target.squeeze(-1) > 0.5
    left_mask = is_reach & (arm > 0.5) & (arm < 1.5)
    right_mask = is_reach & (arm > 1.5)

    left_tgt = self.robot.data.default_joint_pos[:, self._left_ids].clone()
    right_tgt = self.robot.data.default_joint_pos[:, self._right_ids].clone()

    # Anchored world point → current base; residual also in base.
    target_base = world_to_base(self.robot, grasp.target_pos_w)
    desired_base = target_base + self._delta

    if bool(left_mask.any()):
      left_tgt = self._ik_solve(
        env_mask=left_mask,
        joint_ids=self._left_ids,
        body_ids=self._left_bodies,
        ee_body=self._left_ee,
        desired_base=desired_base,
        out_targets=left_tgt,
      )
    if bool(right_mask.any()):
      right_tgt = self._ik_solve(
        env_mask=right_mask,
        joint_ids=self._right_ids,
        body_ids=self._right_bodies,
        ee_body=self._right_ee,
        desired_base=desired_base,
        out_targets=right_tgt,
      )
    self.robot.set_joint_position_target(left_tgt, joint_ids=self._left_ids)
    self.robot.set_joint_position_target(right_tgt, joint_ids=self._right_ids)

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
    # Rotate columns into base: J_base = R_yaw^T J_w
    J_b = torch.zeros_like(J_w)
    for i in range(n_arm):
      J_b[:, :, i] = rotate_world_to_base(self.robot, J_w[:, :, i], ids)
    return J_b

  def _ik_solve(
    self,
    *,
    env_mask: torch.Tensor,
    joint_ids: torch.Tensor,
    body_ids: torch.Tensor,
    ee_body: int,
    desired_base: torch.Tensor,
    out_targets: torch.Tensor,
  ) -> torch.Tensor:
    ids = env_mask.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
      return out_targets
    q = self.robot.data.joint_pos[ids][:, joint_ids].clone()
    desired = desired_base[ids]
    ee_w = self.robot.data.body_link_pos_w[ids, ee_body]
    ee = world_to_base(self.robot, ee_w, ids)
    eye = torch.eye(3, device=self.device, dtype=q.dtype).expand(ids.numel(), 3, 3)
    J = self._geometric_jacobian_base(ids, body_ids, ee_body)

    for _ in range(self._ik_iters):
      err = desired - ee
      JJt = torch.bmm(J, J.transpose(1, 2))
      A = JJt + (self._damping**2) * eye
      delta_x = torch.linalg.solve(A, err.unsqueeze(-1)).squeeze(-1)
      dq = torch.bmm(J.transpose(1, 2), delta_x.unsqueeze(-1)).squeeze(-1)
      dq = torch.clamp(dq, -self._max_dq, self._max_dq)
      q = q + dq
      ee = ee + torch.bmm(J, dq.unsqueeze(-1)).squeeze(-1)

    out_targets = out_targets.clone()
    out_targets[ids] = q
    return out_targets


@dataclass(kw_only=True)
class EeResidualIkActionCfg(ActionTermCfg):
  """Config for 3-D EE residual (base frame) + multi-iter DLS IK."""

  entity_name: str = "robot"
  # Conservative defaults (phase-3 env_cfgs may tighten further).
  ee_delta_scale: tuple[float, float, float] = (0.03, 0.03, 0.05)
  ee_delta_clip: tuple[float, float, float] = (0.04, 0.04, 0.06)
  ik_damping: float = 0.08
  max_joint_delta: float = 0.08
  ik_iterations: int = 4
  fd_eps: float = 1e-3

  def build(self, env: ManagerBasedRlEnv) -> EeResidualIkAction:
    return EeResidualIkAction(self, env)
