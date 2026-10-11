"""Homie waist-grasp target (phase-2 waist-only; arm training disabled).

Uses ``waist_curriculum`` + random grasp points as obs for the waist policy.
``arm_on`` / ``is_target`` stay off so arm reach is not trained (arms PD-held).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

from homierl_mjlab.mdp.curriculum import CurriculumScheduler

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Homie ``g1_inspire_wiast_config.target.ranges``
_DEFAULT_POS_X = (0.15, 0.35)
_DEFAULT_POS_Y = (-0.1, 0.1)
_DEFAULT_POS_Z = (-0.05, 0.1)

_ARM_REACH = 0.49
_LEFT_REF = (0.0, 0.1, 0.29)
_RIGHT_REF = (0.0, -0.1, 0.29)


class GraspTargetCommand(CommandTerm):
  """Waist-only grasp targets (Homie commented ``waist_on`` path)."""

  cfg: GraspTargetCommandCfg

  def __init__(self, cfg: GraspTargetCommandCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.target_position = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.is_target = torch.zeros(self.num_envs, 1, device=self.device)
    self.is_waist = torch.zeros(self.num_envs, 1, device=self.device)
    self.is_target_height = torch.zeros(self.num_envs, 1, device=self.device)
    self.extend_dist = torch.zeros(self.num_envs, 1, device=self.device)
    self.need_waist = torch.zeros(self.num_envs, 1, device=self.device)
    self.is_reachable = torch.zeros(self.num_envs, 1, device=self.device)
    self.target_arm = torch.zeros(self.num_envs, 1, device=self.device)
    self.init_robot_poses = torch.zeros(self.num_envs, 3, device=self.device)

    # Arm curriculum kept for API parity but never enabled (waist-only stage).
    self.arm_curriculum = CurriculumScheduler(
      start_step=50 * 0,
      end_step=50 * 20000,
      min_value=0.5,
      max_value=1.0,
      warmup=50 * 8000,
      ramp=8000,
    )
    # Waist-only stage: start immediately (Homie used start=50*1000).
    self.waist_curriculum = CurriculumScheduler(
      start_step=0,
      end_step=50 * 14000,
      min_value=0.5,
      max_value=0.9,
      warmup=50 * 8000,
      ramp=8000,
    )
    self.arm_on = False
    self.waist_on = False

    self._left_shoulder_id: int | None = None
    self._right_shoulder_id: int | None = None
    for pattern, attr in (
      ("left_shoulder_pitch_link", "_left_shoulder_id"),
      ("right_shoulder_pitch_link", "_right_shoulder_id"),
    ):
      try:
        ids, _ = self.robot.find_bodies(pattern, preserve_order=True)
        if len(ids):
          setattr(self, attr, int(ids[0]))
      except Exception:  # noqa: BLE001
        pass

    self.metrics["grasp_active"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["extend_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["waist_curriculum"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Obs: ``[is_target, is_waist, extend_dist, tx, ty, tz, target_arm]``."""
    return torch.cat(
      (
        self.is_target,
        self.is_waist,
        self.extend_dist,
        self.target_position,
        self.target_arm,
      ),
      dim=-1,
    )

  def _refresh_curriculum(self, *, reshuffle_waist: bool = False) -> None:
    """Waist-only: ``arm_on`` forced off; ``waist_on`` from Homie waist schedule.

    Homie commented waist path::

        if not waist_on: is_waist = 0
        else: is_target = 0; is_waist = (rand < 0.5)

    ``is_waist`` is reshuffled on command resample (not every physics step).
    """
    step = int(self._env.common_step_counter)
    self.arm_on = False  # arm reach training disabled
    self.is_target.zero_()
    prob2 = self.waist_curriculum.value(step)
    was_on = self.waist_on
    self.waist_on = prob2 >= 0.5
    self.metrics["waist_curriculum"][:] = prob2
    if not self.waist_on:
      self.is_waist.zero_()
      return
    if reshuffle_waist or not was_on:
      # Mix loco / waist envs while curriculum is on (Homie commented path).
      rand = torch.rand(self.num_envs, 1, device=self.device)
      self.is_waist[:] = (rand < 0.5).float()

  def _waist_gate(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.is_waist[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Homie ``_resample_target_commands`` gated by ``is_waist`` (not arm)."""
    self._refresh_curriculum(reshuffle_waist=True)
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)

    rx = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_x)
    ry = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_y)
    rz = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_z)
    # Homie older path: xy/z *= is_waist; z = U*gate + 0.3
    self.target_position[env_ids, 0:1] = rx * gate
    self.target_position[env_ids, 1:2] = ry * gate
    self.target_position[env_ids, 2:3] = rz * gate + 0.3

    is_left = self.target_position[env_ids, 1] > 0.0
    self.target_arm[env_ids] = torch.where(
      is_left,
      torch.ones(n, device=self.device),
      torch.full((n,), 2.0, device=self.device),
    ).unsqueeze(-1) * gate

    left_ref = torch.tensor(_LEFT_REF, device=self.device)
    right_ref = torch.tensor(_RIGHT_REF, device=self.device)
    ref = torch.where(is_left.unsqueeze(-1), left_ref, right_ref)
    dist = torch.linalg.norm(self.target_position[env_ids] - ref, dim=-1)
    self.need_waist[env_ids] = ((dist - _ARM_REACH) > 0.0).float().unsqueeze(-1) * gate

    self.init_robot_poses[env_ids] = self.robot.data.root_link_pos_w[env_ids, :3]
    self._refresh_world_target(env_ids)

  def resample_target_z(self, env_ids: torch.Tensor) -> None:
    """Homie target_z sample inside ``_resample_commands`` (gated by is_waist)."""
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)
    rz = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_z)
    self.target_position[env_ids, 2:3] = rz * gate + 0.3
    self._refresh_world_target(env_ids)

  def _refresh_world_target(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return
    root = self.robot.data.root_link_pos_w[env_ids, :3]
    heading = yaw_quat(self.robot.data.root_link_quat_w[env_ids])
    delta = quat_apply(heading, self.target_position[env_ids])
    self.target_pos_w[env_ids, :2] = root[:, :2] + delta[:, :2]
    self.target_pos_w[env_ids, 2] = (
      float(self.cfg.base_height_target) + self.target_position[env_ids, 2]
    )

  def _update_extend_dist(self) -> None:
    active = (self.is_waist.squeeze(-1) > 0).nonzero(as_tuple=False).flatten()
    self.extend_dist.zero_()
    self.is_reachable.zero_()
    if len(active) == 0:
      return
    self._refresh_world_target(active)

    left = active[(self.target_arm[active, 0] == 1)]
    right = active[(self.target_arm[active, 0] == 2)]

    def _shoulder_pos(env_ids: torch.Tensor, body_id: int | None) -> torch.Tensor:
      if body_id is not None:
        return self.robot.data.body_link_pos_w[env_ids, body_id]
      return self.robot.data.root_link_pos_w[env_ids, :3]

    if len(left):
      sh = _shoulder_pos(left, self._left_shoulder_id)
      d = torch.linalg.norm(sh - self.target_pos_w[left], dim=-1, keepdim=True)
      self.extend_dist[left] = torch.clamp(d - _ARM_REACH, min=0.0)
      self.is_reachable[left] = (d < _ARM_REACH).float()
    if len(right):
      sh = _shoulder_pos(right, self._right_shoulder_id)
      d = torch.linalg.norm(sh - self.target_pos_w[right], dim=-1, keepdim=True)
      self.extend_dist[right] = torch.clamp(d - _ARM_REACH, min=0.0)
      self.is_reachable[right] = (d < _ARM_REACH).float()

  def _apply_twist_override(self) -> None:
    """Waist envs: zero vel + stand height (Homie waist mode)."""
    active = self.is_waist.squeeze(-1) > 0.5
    if not torch.any(active):
      return
    try:
      twist = self._env.command_manager.get_term("twist")
    except Exception:  # noqa: BLE001
      return
    twist.vel_command_b[active] = 0.0
    if hasattr(twist, "vel_command_w"):
      twist.vel_command_w[active] = 0.0
    twist.height_command[active] = 0.0
    twist.is_standing_env[active] = True
    if hasattr(twist, "is_height_env"):
      twist.is_height_env[active] = False

  def _update_metrics(self) -> None:
    self.metrics["grasp_active"][:] = self.is_waist.squeeze(-1)
    self.metrics["extend_dist"][:] = self.extend_dist.squeeze(-1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    was_on = self.waist_on
    self._refresh_curriculum(reshuffle_waist=False)
    if self.waist_on and not was_on:
      # Curriculum just flipped on → assign waist mask + targets once.
      self._resample_command(torch.arange(self.num_envs, device=self.device))
    if not self.waist_on:
      self.extend_dist.zero_()
      self.need_waist.zero_()
      self.is_reachable.zero_()
      self.target_arm.zero_()
      self.target_position.zero_()
      self.is_target_height.zero_()
    else:
      off = (self.is_waist.squeeze(-1) < 0.5).nonzero(as_tuple=False).flatten()
      if len(off):
        self.target_position[off] = 0.0
        self.target_arm[off] = 0.0
        self.need_waist[off] = 0.0
        self.extend_dist[off] = 0.0
        self.is_reachable[off] = 0.0
    self._update_extend_dist()
    self._apply_twist_override()


@dataclass(kw_only=True)
class GraspTargetCommandCfg(CommandTermCfg):
  """Config for Homie waist grasp targets (arm training off)."""

  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (4.0, 4.0)
  debug_vis: bool = True
  base_height_target: float = 0.74
  pos_x: tuple[float, float] = _DEFAULT_POS_X
  pos_y: tuple[float, float] = _DEFAULT_POS_Y
  pos_z: tuple[float, float] = _DEFAULT_POS_Z

  def build(self, env: ManagerBasedRlEnv) -> GraspTargetCommand:
    return GraspTargetCommand(self, env)
