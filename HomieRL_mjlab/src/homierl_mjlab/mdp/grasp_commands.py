"""Homie waist-grasp target (phase-2 waist bend; arms via env random curriculum).

Uses ``waist_curriculum`` + random grasp points as obs for the waist policy.
``arm_on`` / ``is_target`` stay off (no arm-reach policy); arm pose comes from
the shared upper-body random curriculum in ``env_cfgs`` events.

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
    self._gui_extend_dist: Any | None = None
    self._gui_shoulder_dist: Any | None = None
    self._gui_need_waist: Any | None = None
    self._gui_waist_pitch_rad: Any | None = None
    self._gui_waist_pitch_deg: Any | None = None
    self._waist_pitch_joint_id: int | None = None
    try:
      ids, _ = self.robot.find_joints("waist_pitch_joint", preserve_order=True)
      if len(ids):
        self._waist_pitch_joint_id = int(ids[0])
    except Exception:  # noqa: BLE001
      pass

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
    self.arm_on = False  # arm reach training disabled
    self.is_target.zero_()
    prob2 = self.waist_curriculum.value(step)
    was_on = self.waist_on
    self.waist_on = prob2 >= 0.5
    self.metrics["waist_curriculum"][:] = prob2
    if not self.waist_on:
      self.is_waist.zero_()
      return
    if not (reshuffle_waist or not was_on):
      return
    if env_ids is None or (reshuffle_waist and not was_on):
      # First time waist turns on: assign mask for everyone once.
      rand = torch.rand(self.num_envs, 1, device=self.device)
      self.is_waist[:] = (rand < 0.5).float()
      return
    # Only reshuffle envs that also get new targets below.
    n = len(env_ids)
    if n == 0:
      return
    rand = torch.rand(n, 1, device=self.device)
    self.is_waist[env_ids] = (rand < 0.5).float()

  def _waist_gate(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.is_waist[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Homie ``_resample_target_commands``: XY (+ arm/need_waist); z from twist."""
    self._refresh_curriculum(reshuffle_waist=True, env_ids=env_ids)
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)

    rx = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_x)
    ry = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_y)
    self.target_position[env_ids, 0:1] = rx * gate
    self.target_position[env_ids, 1:2] = ry * gate
    # Re-apply Homie z after waist reshuffle (twist may have used stale is_waist).
    # is_target_height already implies is_height in Homie bands.
    self.resample_target_z(env_ids)
    self.init_robot_poses[env_ids] = self.robot.data.root_link_pos_w[env_ids, :3]

  def _finalize_waist_targets(
    self, env_ids: torch.Tensor, gate: torch.Tensor
  ) -> None:
    """Arm side + ``need_waist`` (does not own ``is_target_height``)."""
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
    # Homie default: 3D to standing shoulder. Low target_z alone then forces
    # need_waist → bend reward while height rewards ask for squat.
    # When ``is_target_height`` (height squat path): XY-only so squat is not
    # mislabeled as bend; otherwise keep Homie 3D for far horizontal reach.
    dist_3d = torch.linalg.norm(self.target_position[env_ids, 0:3] - ref, dim=-1)
    dist_xy = torch.linalg.norm(
      self.target_position[env_ids, 0:2] - ref[:, 0:2], dim=-1
    )
    ith = self.is_target_height[env_ids].squeeze(-1) > 0.5
    dist = torch.where(ith, dist_xy, dist_3d)
    self.need_waist[env_ids] = ((dist - _ARM_REACH) > 0.0).float().unsqueeze(-1) * gate

  def resample_target_z(
    self,
    env_ids: torch.Tensor,
    is_height: torch.Tensor | None = None,
    is_target_height: torch.Tensor | None = None,
  ) -> None:
    """Homie ``target_z = U(pos_z)*is_height*is_target_height*is_waist + 0.3``.

    Two squat paths (Homie ``_resample_commands``):
    - height-cmd: ``is_height & is_waist & ~is_target_height`` → z stays 0.3,
      height offset on twist.
    - target-height: ``is_height & is_target_height & is_waist`` → sample z;
      ``h_ref`` from ``target_z`` in height rewards.
    """
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)
    rz = torch.empty(n, 1, device=self.device).uniform_(*self.cfg.pos_z)
    if is_height is not None and is_target_height is not None:
      h = is_height.float().view(n, 1)
      ith = is_target_height.float().view(n, 1)
      self.target_position[env_ids, 2:3] = rz * h * ith * gate + 0.3
    elif is_target_height is not None:
      ith = is_target_height.float().view(n, 1)
      self.target_position[env_ids, 2:3] = rz * ith * gate + 0.3
    else:
      ith = self.is_target_height[env_ids]
      self.target_position[env_ids, 2:3] = rz * ith * gate + 0.3
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
    # Play: no twist band — treat non-neutral z as Homie ``is_target_height``.
    self.is_target_height[idx] = 1.0 if abs(z - 0.3) > 1e-3 else 0.0
    self.target_position[idx, 0] = x
    self.target_position[idx, 1] = y
    self.target_position[idx, 2] = z
    self.target_arm[idx] = 1.0 if y > 0.0 else 2.0
    ref = _LEFT_REF if y > 0.0 else _RIGHT_REF
    # Match train: target-height (non-neutral z) → XY need_waist; else 3D.
    if abs(z - 0.3) > 1e-3:
      dist = float(np.hypot(x - ref[0], y - ref[1]))
    else:
      dist = float(np.linalg.norm([x - ref[0], y - ref[1], z - ref[2]]))
    self.need_waist[idx] = 1.0 if (dist - _ARM_REACH) > 0.0 else 0.0
    env_ids = torch.tensor([idx], device=self.device, dtype=torch.long)
    self._refresh_world_target(env_ids)
    self._update_extend_dist()
    self._apply_twist_override()
    # Target-height squat: reward h_ref = base+z+0.10 via _command_height_target.
    if z <= 0.0:
      try:
        twist = self._env.command_manager.get_term("twist")
        # Height obs stays at base when ith; do not fold h_ref into height_cmd.
        twist.height_command[idx] = 0.0
        twist.is_height_env[idx] = False
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

  def _shoulder_target_dist(self, idx: int) -> float:
    """Live ‖shoulder_w − target_w‖ for the viewed env (Homie ``dist``)."""
    arm = float(self.target_arm[idx, 0].item())
    body_id = None
    if arm == 1.0:
      body_id = self._left_shoulder_id
    elif arm == 2.0:
      body_id = self._right_shoulder_id
    if body_id is None:
      sh = self.robot.data.root_link_pos_w[idx]
    else:
      sh = self.robot.data.body_link_pos_w[idx, body_id]
    tgt = self.target_pos_w[idx]
    return float(torch.linalg.norm(sh - tgt).item())

  def _sync_gui_readouts(self) -> None:
    """Push live extend_dist / shoulder dist / waist_pitch (rad+deg) to Viser."""
    if self._gui_get_env_idx is None:
      return
    if (
      self._gui_extend_dist is None
      and self._gui_shoulder_dist is None
      and self._gui_waist_pitch_rad is None
      and self._gui_waist_pitch_deg is None
    ):
      return
    idx = int(self._gui_get_env_idx())
    d = self._shoulder_target_dist(idx)
    if self._gui_shoulder_dist is not None:
      self._gui_shoulder_dist.value = d
    if self._gui_extend_dist is not None:
      # Same as Homie: clamp(dist - 0.49, min=0). Not ‖pelvis→target‖.
      self._gui_extend_dist.value = float(self.extend_dist[idx, 0].item())
    if self._waist_pitch_joint_id is not None:
      q = float(self.robot.data.joint_pos[idx, self._waist_pitch_joint_id].item())
      if self._gui_waist_pitch_rad is not None:
        self._gui_waist_pitch_rad.value = q
      if self._gui_waist_pitch_deg is not None:
        self._gui_waist_pitch_deg.value = q * (180.0 / np.pi)

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

    # Body-frame ranges (match obs). Train: z = U(pos_z)+0.3 only when
    # is_height & is_target_height & is_waist; else 0.3 (height-cmd squat).
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
        hint="z=0.3: height-cmd squat if Twist height. Else target-height path; low z → h_ref + 3D need_waist.",
      )
      shoulder_readout = server.gui.add_number(
        "shoulder→target (m)",
        initial_value=0.0,
        step=0.001,
        disabled=True,
        hint="Live ‖shoulder_pitch_link − target_w‖. target_x=0.65 is from pelvis, not shoulder.",
      )
      extend_readout = server.gui.add_number(
        "extend_dist (m)",
        initial_value=0.0,
        step=0.001,
        disabled=True,
        hint="Homie: max(0, shoulder→target − 0.49).",
      )
      pitch_rad = server.gui.add_number(
        "waist_pitch (rad)",
        initial_value=0.0,
        step=0.001,
        disabled=True,
        hint="Live waist_pitch_joint in radians.",
      )
      pitch_deg = server.gui.add_number(
        "waist_pitch (deg)",
        initial_value=0.0,
        step=0.1,
        disabled=True,
        hint="Live waist_pitch_joint in degrees.",
      )
      reset_btn = server.gui.add_button("Reset target", icon=Icon.REFRESH)

      @reset_btn.on_click
      def _(_) -> None:
        sx.value = x0
        sy.value = y0
        sz.value = z0

    self._gui_enabled = enabled
    self._gui_sliders = [sx, sy, sz]
    self._gui_shoulder_dist = shoulder_readout
    self._gui_extend_dist = extend_readout
    self._gui_waist_pitch_rad = pitch_rad
    self._gui_waist_pitch_deg = pitch_deg
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
    self._sync_gui_readouts()

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
