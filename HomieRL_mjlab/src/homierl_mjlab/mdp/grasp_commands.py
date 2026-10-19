"""Homie waist-grasp target (phase-2 waist-only; arm training disabled).

Uses ``waist_curriculum`` + random grasp points as obs for the waist policy.
``arm_on`` / ``is_target`` stay off so arm reach is not trained (arms PD-held).

Viser play: ``create_gui`` exposes target_x/y/z so you can drive waist mode
manually (Enable ON → force ``is_waist`` + stand still on the viewed env).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

from homierl_mjlab.mdp.curriculum import CurriculumScheduler

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Homie waist-bend training ranges (``g1_inspire_wiast_config1.target.ranges``).
# Narrow paper defaults (0.15..0.35 / -0.05..0.1) never exceed arm reach 0.49
# → ``need_waist`` stays 0 → ``deviation_waist_joint`` only has zero_loss (keep upright).
_DEFAULT_POS_X = (0.05, 0.65)
_DEFAULT_POS_Y = (-0.35, 0.35)
_DEFAULT_POS_Z = (-1.0, 0.2)

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
      (".*left_shoulder_pitch_link", "_left_shoulder_id"),
      (".*right_shoulder_pitch_link", "_right_shoulder_id"),
    ):
      try:
        ids, _ = self.robot.find_bodies(pattern, preserve_order=True)
        if len(ids):
          setattr(self, attr, int(ids[0]))
      except Exception:  # noqa: BLE001
        pass

    self.metrics["grasp_active"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["extend_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["need_waist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["waist_curriculum"] = torch.zeros(self.num_envs, device=self.device)

    # Viser play GUI (set by create_gui).
    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_sliders: list[viser.GuiSliderHandle] = []
    self._gui_get_env_idx: Callable[[], int] | None = None

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

  def _refresh_curriculum(
    self,
    *,
    reshuffle_waist: bool = False,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Waist-only: ``arm_on`` forced off; ``waist_on`` from Homie waist schedule.

    Homie commented waist path::

        if not waist_on: is_waist = 0
        else: is_target = 0; is_waist = (rand < 0.5)

    Important: when ``reshuffle_waist``, only shuffle ``env_ids`` (the envs
    being target-resampled). Reshuffling *all* envs while only writing targets
    for a subset leaves new waist envs with zero targets → ``need_waist=0``
    forever (TB: need_waist/extend_dist ≈ 0).
    """
    step = int(self._env.common_step_counter)
    # Phase-3: arm reach on waist envs; phase-2 keeps arm_on off.
    self.arm_on = bool(self.cfg.enable_arm_reach)
    prob2 = self.waist_curriculum.value(step)
    was_on = self.waist_on
    # Phase-3 always allows waist/loco mix via is_waist_prob (not Homie schedule).
    self.waist_on = True if self.arm_on else (prob2 >= 0.5)
    self.metrics["waist_curriculum"][:] = (
      float(self.cfg.is_waist_prob) if self.arm_on else prob2
    )
    if not self.waist_on:
      self.is_waist.zero_()
      self.is_target.zero_()
      return
    if not (reshuffle_waist or not was_on):
      if self.arm_on:
        self.is_target[:] = self.is_waist
      return
    waist_p = float(self.cfg.is_waist_prob) if self.arm_on else 0.5
    if env_ids is None or (reshuffle_waist and not was_on):
      rand = torch.rand(self.num_envs, 1, device=self.device)
      self.is_waist[:] = (rand < waist_p).float()
      self.is_target[:] = self.is_waist if self.arm_on else torch.zeros_like(self.is_waist)
      return
    n = len(env_ids)
    if n == 0:
      return
    rand = torch.rand(n, 1, device=self.device)
    self.is_waist[env_ids] = (rand < waist_p).float()
    if self.arm_on:
      self.is_target[env_ids] = self.is_waist[env_ids]
    else:
      self.is_target[env_ids] = 0.0

  def _waist_gate(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.is_waist[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Homie ``_resample_target_commands`` gated by ``is_waist`` (not arm)."""
    self._refresh_curriculum(reshuffle_waist=True, env_ids=env_ids)
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)

    rx = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_x)
    ry = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_y)
    # Waist envs: sample full xyz (incl. low z) so need_waist / bend can fire.
    # Homie xy-only + z from height band left most waist targets reachable →
    # extend_dist≈0 and deviation_waist only kept upright (zero_loss).
    rz = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_z)
    self.target_position[env_ids, 0:1] = rx * gate
    self.target_position[env_ids, 1:2] = ry * gate
    self.target_position[env_ids, 2:3] = rz * gate + 0.3
    self._finalize_waist_targets(env_ids, gate)

    self.init_robot_poses[env_ids] = self.robot.data.root_link_pos_w[env_ids, :3]
    self._refresh_world_target(env_ids)

  def _finalize_waist_targets(
    self, env_ids: torch.Tensor, gate: torch.Tensor
  ) -> None:
    """Arm side, need_waist, and is_target_height from body-frame target."""
    n = len(env_ids)
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
    # Low / non-neutral z → squat h_ref path in height rewards.
    tz = self.target_position[env_ids, 2:3]
    self.is_target_height[env_ids] = (
      (torch.abs(tz - 0.3) > 1e-3).float() * gate
    )

  def resample_target_z(
    self,
    env_ids: torch.Tensor,
    is_height: torch.Tensor | None = None,
    is_target_height: torch.Tensor | None = None,
  ) -> None:
    """Refresh target_z.

    Waist envs: always full ``pos_z`` range (bend curriculum).
    Non-waist: Homie ``U(pos_z)*is_height*is_target_height+0.3`` when flags given.
    """
    del is_target_height  # waist path owns height flag via _finalize_waist_targets
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)
    rz = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_z)
    z_waist = rz + 0.3
    if is_height is not None:
      h = is_height.float().view(n, 1)
      # Non-waist keep Homie neutral z=0.3 unless height band (unused for bend).
      z_other = 0.3 * torch.ones(n, 1, device=self.device)
      self.target_position[env_ids, 2:3] = torch.where(gate > 0.5, z_waist, z_other)
    else:
      self.target_position[env_ids, 2:3] = z_waist * gate + 0.3 * (1.0 - gate)
    self._finalize_waist_targets(env_ids, gate)
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
    """Waist envs: zero velocity only (Homie ``commands *= (1-is_waist)``).

    Do **not** wipe height — Homie still samples squat offsets on waist
    (``is_height * is_waist * (1-is_target_height)``) and tracks ``h_ref`` from
    low ``target_z`` when ``is_target_height``.
    """
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
    twist.is_standing_env[active] = True

  def _update_metrics(self) -> None:
    self.metrics["grasp_active"][:] = self.is_waist.squeeze(-1)
    self.metrics["extend_dist"][:] = self.extend_dist.squeeze(-1)
    self.metrics["need_waist"][:] = self.need_waist.squeeze(-1)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    # Viser play: Enable ON → waist target owns viewed env (zeros vel).
    # Enable OFF → release viewed env so Twist joystick/height work.
    if self._gui_enabled is not None:
      if self._gui_active():
        self._apply_gui_target()
        return
      self._release_gui_env()
      # Play with GUI present: do not keep random is_waist fighting Twist.
      return
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

  def _gui_active(self) -> bool:
    return self._gui_enabled is not None and bool(self._gui_enabled.value)

  def _release_gui_env(self) -> None:
    """Clear waist flags on the viewed env so Twist commands are not zeroed."""
    if self._gui_get_env_idx is None:
      return
    idx = int(self._gui_get_env_idx())
    self.is_waist[idx] = 0.0
    self.is_target[idx] = 0.0
    self.is_target_height[idx] = 0.0
    self.target_position[idx] = 0.0
    self.target_arm[idx] = 0.0
    self.need_waist[idx] = 0.0
    self.extend_dist[idx] = 0.0
    self.is_reachable[idx] = 0.0

  def _apply_manual_target(self, idx: int, x: float, y: float, z: float) -> None:
    """Force waist mode + body-frame target on one env (Viser play)."""
    self.waist_on = True
    self.is_waist[idx] = 1.0
    self.is_target[idx] = 0.0
    # Low / non-neutral z → Homie ``is_target_height`` path (squat h_ref).
    self.is_target_height[idx] = 1.0 if abs(z - 0.3) > 1e-3 else 0.0
    self.target_position[idx, 0] = x
    self.target_position[idx, 1] = y
    self.target_position[idx, 2] = z
    self.target_arm[idx] = 1.0 if y > 0.0 else 2.0
    ref = _LEFT_REF if y > 0.0 else _RIGHT_REF
    dist = float(
      np.linalg.norm(
        np.array([x, y, z], dtype=np.float64) - np.array(ref, dtype=np.float64)
      )
    )
    self.need_waist[idx] = 1.0 if (dist - _ARM_REACH) > 0.0 else 0.0
    env_ids = torch.tensor([idx], device=self.device, dtype=torch.long)
    self._refresh_world_target(env_ids)
    self._update_extend_dist()
    self._apply_twist_override()
    # Mirror Homie squat when target is low.
    if z < 0.0:
      try:
        twist = self._env.command_manager.get_term("twist")
        h_ref = float(np.clip(z + 0.64, 0.25, 0.74))
        twist.height_command[idx] = h_ref - float(self.cfg.base_height_target)
        twist.is_height_env[idx] = True
        twist.is_standing_env[idx] = True
      except Exception:  # noqa: BLE001
        pass

  def _apply_gui_target(self) -> None:
    if self._gui_get_env_idx is None or len(self._gui_sliders) < 3:
      return
    idx = int(self._gui_get_env_idx())
    x = float(self._gui_sliders[0].value)
    y = float(self._gui_sliders[1].value)
    z = float(self._gui_sliders[2].value)
    self._apply_manual_target(idx, x, y, z)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Viser: Enable + body-frame target_x/y/z for waist play."""
    del on_change, request_action
    from viser import Icon

    # Body-frame ranges (match obs). Train stores z as U(pos_z)+0.3 when
    # is_target_height; play exposes the full stored range so need_waist can fire.
    x_lo, x_hi = float(self.cfg.pos_x[0]), float(self.cfg.pos_x[1])
    y_lo, y_hi = float(self.cfg.pos_y[0]), float(self.cfg.pos_y[1])
    z_lo = float(self.cfg.pos_z[0]) + 0.3  # e.g. -0.7
    z_hi = float(self.cfg.pos_z[1]) + 0.3  # e.g. 0.5
    # Defaults ready for waist test; panel starts OFF so Twist loco/height works.
    x0 = 0.45
    y0 = 0.10
    z0 = -0.20

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox(
        "Enable waist target",
        initial_value=False,
        hint="OFF=loco (Twist). ON=waist mode: zeros vel, drives body-frame target.",
      )
      sx = server.gui.add_slider(
        "target_x (body)",
        min=x_lo,
        max=x_hi,
        step=0.01,
        initial_value=x0,
      )
      sy = server.gui.add_slider(
        "target_y (body)",
        min=y_lo,
        max=y_hi,
        step=0.01,
        initial_value=y0,
      )
      sz = server.gui.add_slider(
        "target_z (body, obs)",
        min=z_lo,
        max=z_hi,
        step=0.01,
        initial_value=z0,
        hint="z<0 → squat h_ref; far/low → need_waist for bend reward.",
      )
      reset_btn = server.gui.add_button("Reset target", icon=Icon.REFRESH)

      @reset_btn.on_click
      def _(_) -> None:
        sx.value = x0
        sy.value = y0
        sz.value = z0

    self._gui_enabled = enabled
    self._gui_sliders = [sx, sy, sz]
    self._gui_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    # After Twist joystick: waist Enable ON zeros vel; Enable OFF releases env.
    if self._gui_enabled is None:
      return
    if self._gui_active():
      self._apply_gui_target()
    else:
      self._release_gui_env()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw grasp target sphere (world) when waist is active."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    targets = self.target_pos_w.detach().cpu().numpy()
    is_w = self.is_waist.detach().cpu().numpy().reshape(-1)
    for batch in env_indices:
      if is_w[batch] < 0.5:
        continue
      center = targets[batch]
      if np.linalg.norm(center) < 1e-6:
        continue
      visualizer.add_sphere(
        center,
        radius=0.04,
        color=(1.0, 0.25, 0.15, 0.85),
        label=f"grasp_{batch}",
      )


@dataclass(kw_only=True)
class GraspTargetCommandCfg(CommandTermCfg):
  """Config for Homie waist grasp targets (arm training off by default)."""

  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (4.0, 4.0)
  debug_vis: bool = True
  base_height_target: float = 0.74
  pos_x: tuple[float, float] = _DEFAULT_POS_X
  pos_y: tuple[float, float] = _DEFAULT_POS_Y
  pos_z: tuple[float, float] = _DEFAULT_POS_Z
  # Phase-3: enable arm reach (is_target) for EE residual + IK.
  enable_arm_reach: bool = False
  # Phase-3 mix: fraction of envs with is_waist=1 (rest keep loco-only).
  is_waist_prob: float = 0.25

  def build(self, env: ManagerBasedRlEnv) -> GraspTargetCommand:
    return GraspTargetCommand(self, env)
