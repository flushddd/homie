"""Homie waist-grasp target (phase-2 bend; phase-3 loco/waist/reach mix).

Phase-2: ``waist_curriculum`` + ``is_waist_prob``; ``is_target`` stays 0 (arms held).
Phase-3: ``enable_arm_reach`` + ternary mode mix
  ``(p_loco, p_waist_only, p_reach)`` → flags ``(0,0) / (1,0) / (1,1)``.
  Reach share and target XY/Z ranges curriculum from ``*_start`` → final.

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

from homierl_mjlab.mdp.base_frame import base_to_world, world_to_base
from homierl_mjlab.mdp.curriculum import CurriculumScheduler

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Homie waist-bend training ranges (``g1_inspire_wiast_config1.target.ranges``).
# Narrow paper defaults (0.15..0.35 / -0.05..0.1) never exceed arm reach 0.49
# → ``need_waist`` stays 0 → ``deviation_waist_joint`` only has zero_loss (keep upright).
# Phase-2 uses these full ranges as before; phase-3 curricula start→end.
_DEFAULT_POS_X = (0.05, 0.65)
_DEFAULT_POS_Y = (-0.35, 0.35)
_DEFAULT_POS_Z = (-1.0, 0.2)
# Phase-3 curriculum starts closer / shallower, then expands to the defaults.
_START_POS_X = (0.15, 0.40)
_START_POS_Y = (-0.18, 0.18)
_START_POS_Z = (-0.45, 0.10)
# Small reach from iter 0 (IK on ~10% envs), then ramp up — not a zero-IK warmup.
_START_MODE_PROBS = (0.45, 0.45, 0.10)  # loco / waist_only / reach
_END_MODE_PROBS = (0.2, 0.2, 0.6)

_ARM_REACH = 0.49
_LEFT_REF = (0.0, 0.1, 0.29)
_RIGHT_REF = (0.0, -0.1, 0.29)


def _lerp_pair(
  start: tuple[float, float], end: tuple[float, float], alpha: float
) -> tuple[float, float]:
  a = float(alpha)
  return (
    start[0] + a * (end[0] - start[0]),
    start[1] + a * (end[1] - start[1]),
  )


def _lerp_triple(
  start: tuple[float, float, float],
  end: tuple[float, float, float],
  alpha: float,
) -> tuple[float, float, float]:
  a = float(alpha)
  return (
    start[0] + a * (end[0] - start[0]),
    start[1] + a * (end[1] - start[1]),
    start[2] + a * (end[2] - start[2]),
  )


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
    # Phase-3: 0→1 progress for mode mix + target range expansion.
    self.reach_mode_curriculum = CurriculumScheduler(
      start_step=0,
      end_step=int(cfg.mode_curriculum_ramp) + int(cfg.mode_curriculum_warmup),
      min_value=0.0,
      max_value=1.0,
      warmup=int(cfg.mode_curriculum_warmup),
      ramp=int(cfg.mode_curriculum_ramp),
    )
    self.target_range_curriculum = CurriculumScheduler(
      start_step=0,
      end_step=int(cfg.target_range_ramp) + int(cfg.target_range_warmup),
      min_value=0.0,
      max_value=1.0,
      warmup=int(cfg.target_range_warmup),
      ramp=int(cfg.target_range_ramp),
    )
    self.arm_on = False
    self.waist_on = False
    # Latch on first use so phase-2 resume common_step_counter does not
    # instantly max out reach/target curriculum (was alpha=1 at iter 0).
    self._curriculum_t0: int | None = None

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
    self.metrics["reach_mode_alpha"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_range_alpha"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reach_prob"] = torch.zeros(self.num_envs, device=self.device)

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

  def _phase3_local_step(self) -> int:
    """Env steps since phase-3 curriculum started (not inherited phase-2 counter)."""
    step = int(self._env.common_step_counter)
    if self._curriculum_t0 is None:
      self._curriculum_t0 = step
    return max(0, step - self._curriculum_t0)

  def _mode_progress(self) -> float:
    return float(self.reach_mode_curriculum.value(self._phase3_local_step()))

  def _range_progress(self) -> float:
    return float(self.target_range_curriculum.value(self._phase3_local_step()))

  def _normalized_mode_probs(self) -> tuple[float, float, float]:
    """Phase-3 mix: ``(p_loco, p_waist_only, p_reach)``, curriculum + renormalized."""
    if self.arm_on:
      p = _lerp_triple(self.cfg.mode_probs_start, self.cfg.mode_probs, self._mode_progress())
    else:
      p = self.cfg.mode_probs
    a, b, c = float(p[0]), float(p[1]), float(p[2])
    s = a + b + c
    if s <= 0.0:
      return _END_MODE_PROBS
    return a / s, b / s, c / s

  def _current_pos_ranges(self) -> tuple[tuple[float, float], ...]:
    """XY/Z sample ranges.

    Phase-2: fixed full ``pos_*`` (same as prior phase-2 / config1).
    Phase-3: start (near) → end (far) via target-range curriculum.
    """
    if not self.arm_on:
      return self.cfg.pos_x, self.cfg.pos_y, self.cfg.pos_z
    alpha = self._range_progress()
    return (
      _lerp_pair(self.cfg.pos_x_start, self.cfg.pos_x, alpha),
      _lerp_pair(self.cfg.pos_y_start, self.cfg.pos_y, alpha),
      _lerp_pair(self.cfg.pos_z_start, self.cfg.pos_z, alpha),
    )

  def _sample_mode_flags(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(is_waist, is_target)`` for ``n`` envs.

    Phase-2 (``arm_on=False``): ``is_waist ~ Bern(is_waist_prob)``, ``is_target=0``.
    Phase-3: ternary modes from curriculum ``mode_probs`` —
      loco | waist_only | reach  →  (0,0) | (1,0) | (1,1).
    """
    if not self.arm_on:
      waist_p = float(self.cfg.is_waist_prob)
      is_waist = (torch.rand(n, 1, device=self.device) < waist_p).float()
      return is_waist, torch.zeros_like(is_waist)

    p_loco, p_wo, _p_reach = self._normalized_mode_probs()
    u = torch.rand(n, device=self.device)
    # 0=loco, 1=waist_only, 2=reach
    mode = torch.zeros(n, dtype=torch.long, device=self.device)
    mode = torch.where(u >= p_loco, torch.ones_like(mode), mode)
    mode = torch.where(u >= (p_loco + p_wo), torch.full_like(mode, 2), mode)
    is_waist = (mode >= 1).float().unsqueeze(-1)
    is_target = (mode == 2).float().unsqueeze(-1)
    return is_waist, is_target

  def _refresh_curriculum(
    self,
    *,
    reshuffle_waist: bool = False,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Assign waist / reach mode flags.

    Phase-2: Homie waist schedule + ``is_waist_prob`` (``is_target`` stays 0).
    Phase-3: always on; ternary mix curriculum ``mode_probs_start`` → ``mode_probs``.

    Important: when ``reshuffle_waist``, only shuffle ``env_ids`` (the envs
    being target-resampled). Reshuffling *all* envs while only writing targets
    for a subset leaves new waist envs with zero targets → ``need_waist=0``
    forever (TB: need_waist/extend_dist ≈ 0).
    """
    step = int(self._env.common_step_counter)
    # Phase-3: arm reach enabled; phase-2 keeps arm_on off.
    self.arm_on = bool(self.cfg.enable_arm_reach)
    prob2 = self.waist_curriculum.value(step)
    was_on = self.waist_on
    # Phase-3 always allows mix; phase-2 follows Homie waist curriculum gate.
    self.waist_on = True if self.arm_on else (prob2 >= 0.5)
    mode_alpha = self._mode_progress() if self.arm_on else 0.0
    range_alpha = self._range_progress() if self.arm_on else 0.0
    self.metrics["reach_mode_alpha"][:] = mode_alpha
    self.metrics["target_range_alpha"][:] = range_alpha
    if self.arm_on:
      _pl, _pw, pr = self._normalized_mode_probs()
      self.metrics["waist_curriculum"][:] = pr  # log reach share
      self.metrics["reach_prob"][:] = pr
    else:
      self.metrics["waist_curriculum"][:] = prob2
      self.metrics["reach_prob"].zero_()
    if not self.waist_on:
      self.is_waist.zero_()
      self.is_target.zero_()
      return
    # Keep existing mode flags until the next resample (do not force is_target=is_waist).
    if not (reshuffle_waist or not was_on):
      return
    if env_ids is None or (reshuffle_waist and not was_on):
      is_w, is_t = self._sample_mode_flags(self.num_envs)
      self.is_waist[:] = is_w
      self.is_target[:] = is_t
      return
    n = len(env_ids)
    if n == 0:
      return
    is_w, is_t = self._sample_mode_flags(n)
    self.is_waist[env_ids] = is_w
    self.is_target[env_ids] = is_t

  def _waist_gate(self, env_ids: torch.Tensor) -> torch.Tensor:
    return self.is_waist[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Homie ``_resample_target_commands`` gated by ``is_waist`` (not arm)."""
    self._refresh_curriculum(reshuffle_waist=True, env_ids=env_ids)
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    gate = self._waist_gate(env_ids)

    pos_x, pos_y, pos_z = self._current_pos_ranges()
    rx = torch.empty(n, 1, device=self.device).uniform_(*pos_x)
    ry = torch.empty(n, 1, device=self.device).uniform_(*pos_y)
    # Waist envs: sample xyz (incl. low z) so need_waist / bend can fire.
    # Ranges expand via target-range curriculum (near → far).
    rz = torch.empty(n, 1, device=self.device).uniform_(*pos_z)
    self.target_position[env_ids, 0:1] = rx * gate
    self.target_position[env_ids, 1:2] = ry * gate
    self.target_position[env_ids, 2:3] = rz * gate + 0.3
    self._finalize_waist_targets(env_ids, gate)

    self.init_robot_poses[env_ids] = self.robot.data.root_link_pos_w[env_ids, :3]
    self._freeze_anchor_from_base(env_ids)

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
    _px, _py, pos_z = self._current_pos_ranges()
    rz = torch.empty(n, 1, device=self.device).uniform_(*pos_z)
    z_waist = rz + 0.3
    if is_height is not None:
      h = is_height.float().view(n, 1)
      # Non-waist keep Homie neutral z=0.3 unless height band (unused for bend).
      z_other = 0.3 * torch.ones(n, 1, device=self.device)
      self.target_position[env_ids, 2:3] = torch.where(gate > 0.5, z_waist, z_other)
    else:
      self.target_position[env_ids, 2:3] = z_waist * gate + 0.3 * (1.0 - gate)
    self._finalize_waist_targets(env_ids, gate)
    # Freeze world/task anchor once from the sampled base target.
    self._freeze_anchor_from_base(env_ids)

  def _freeze_anchor_from_base(self, env_ids: torch.Tensor | None = None) -> None:
    """Resample-time: set ``target_pos_w`` from current base sample (fixed until next resample).

    Uses full pelvis + yaw (``base_to_world``), so the physical grasp point stays
    fixed while the robot later squats/bends; obs reprojects via
    ``_reproject_base_from_anchor``.
    """
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return
    self.target_pos_w[env_ids] = base_to_world(
      self.robot, self.target_position[env_ids], env_ids
    )

  def _reproject_base_from_anchor(self, env_ids: torch.Tensor | None = None) -> None:
    """Every step: refresh base-frame ``target_position`` from frozen ``target_pos_w``."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return
    self.target_position[env_ids] = world_to_base(
      self.robot, self.target_pos_w[env_ids], env_ids
    )

  def _update_extend_dist(self) -> None:
    active = (self.is_waist.squeeze(-1) > 0).nonzero(as_tuple=False).flatten()
    self.extend_dist.zero_()
    self.is_reachable.zero_()
    if len(active) == 0:
      return
    # Keep world anchor fixed; update base obs for squat/bend compensation.
    self._reproject_base_from_anchor(active)

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
    self._freeze_anchor_from_base(env_ids)
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
  # Phase-2: fraction of envs with is_waist=1 (is_target stays 0).
  is_waist_prob: float = 0.5
  # Phase-3 ternary mix (loco, waist_only, reach); curriculum start → end.
  # waist_only = bend without EE; reach = bend + EE residual.
  mode_probs_start: tuple[float, float, float] = _START_MODE_PROBS
  mode_probs: tuple[float, float, float] = _END_MODE_PROBS
  # Env-steps: no zero-IK hold; alpha 0→1 over ramp (reach 0.1→0.6).
  mode_curriculum_warmup: int = 0
  mode_curriculum_ramp: int = 50 * 8000  # ~8k iters to full reach share
  # Target box curriculum: near/shallow → full pos_* ranges.
  pos_x_start: tuple[float, float] = _START_POS_X
  pos_y_start: tuple[float, float] = _START_POS_Y
  pos_z_start: tuple[float, float] = _START_POS_Z
  target_range_warmup: int = 0
  target_range_ramp: int = 50 * 10000

  def build(self, env: ManagerBasedRlEnv) -> GraspTargetCommand:
    return GraspTargetCommand(self, env)
