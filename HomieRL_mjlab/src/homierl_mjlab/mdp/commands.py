"""HOMIE twist + height command for mjlab."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TwistHeightCommand(UniformVelocityCommand):
  """Velocity twist command with an extra base-height channel.

  ``command`` shape is ``(num_envs, 4)``: ``[lin_vel_x, lin_vel_y, ang_vel_z, height]``.

  Height is an **offset** relative to ``base_height_target`` (default 0.74 m):
  ``h_target = base_height_target + height_offset``. To squat to absolute height
  ``h``, set ``height_offset = h - base_height_target``.
  """

  cfg: TwistHeightCommandCfg

  def __init__(self, cfg: TwistHeightCommandCfg, env: ManagerBasedRlEnv):
    # Bypass UniformVelocityCommand.__init__ body that assumes 3-D command;
    # call CommandTerm then re-init velocity buffers like the parent.
    CommandTerm.__init__(self, cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.height_command = torch.zeros(self.num_envs, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)
    self.is_forward_env = torch.zeros_like(self.is_heading_env)
    # Homie-style: height-only envs (vel forced 0, height offset sampled).
    self.is_height_env = torch.zeros_like(self.is_heading_env)

    self.commanded_displacement_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.episode_start_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._height_abs_slider: viser.GuiSliderHandle | None = None
    self._joystick_get_env_idx: Callable[[], int] | None = None
    self._gui_waist_pitch_rad: Any = None
    self._gui_waist_pitch_deg: Any = None
    self._gui_waist_pitch_act: Any = None
    self._waist_pitch_jid: int | None = None
    try:
      jnames = list(self.robot.joint_names)
      if "waist_pitch_joint" in jnames:
        self._waist_pitch_jid = jnames.index("waist_pitch_joint")
    except Exception:  # noqa: BLE001
      self._waist_pitch_jid = None

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [self.vel_command_b, self.height_command.unsqueeze(-1)], dim=-1
    )

  def _joystick_active(self) -> bool:
    return self._joystick_enabled is not None and bool(self._joystick_enabled.value)

  def _height_gui_active(self) -> bool:
    """Viser height slider is present — take over height from random/standing logic."""
    return self._height_abs_slider is not None

  def _get_grasp(self):
    """Optional phase-2 grasp term (Homie is_target / is_waist)."""
    try:
      return self._env.command_manager.get_term("grasp")
    except Exception:  # noqa: BLE001
      return None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)

    # Homie ``_resample_commands`` (legged_robot):
    #   set_x < 1/3  → height band
    #   set_x > 1/2  → velocity band
    #   else         → idle
    #   set_y < 1/3  → model_h; else model_target → is_target_height
    # Vel/height further gated by (1-is_target)*(1-is_waist) / is_waist.
    n = len(env_ids)
    if n == 0:
      return
    set_x = torch.rand(n, device=self.device)
    set_y = torch.rand(n, device=self.device)
    is_height = set_x < (1.0 / 3.0)
    is_vel = set_x > 0.5
    model_h = set_y < (1.0 / 3.0)
    model_target = ~model_h
    is_target_height = model_target & is_height

    grasp = self._get_grasp()
    if grasp is not None:
      grasp._refresh_curriculum()
      # Keep Homie buffers in sync for height gating / obs.
      grasp.is_target_height[env_ids] = is_target_height.float().unsqueeze(-1)
      is_t = grasp.is_target[env_ids].squeeze(-1) > 0.5
      is_w = grasp.is_waist[env_ids].squeeze(-1) > 0.5
      # Homie: vel *= is_vel * (1-is_target) * (1-is_waist)
      allow_vel = is_vel & (~is_t) & (~is_w)
      self.is_standing_env[env_ids] = ~allow_vel
      self.is_height_env[env_ids] = False
      self.vel_command_b[env_ids[~allow_vel]] = 0.0
      self.vel_command_w[env_ids[~allow_vel]] = 0.0
      # Homie heading_command=False height:
      #   offset * is_height * (1-is_target) * is_waist * (1-is_target_height)
      # Stored as offset (abs = base + offset).
      r = torch.empty(n, device=self.device)
      h_off = r.uniform_(*self.cfg.ranges.height)
      allow_h = (
        is_height & (~is_t) & is_w & (~is_target_height)
      )
      self.height_command[env_ids] = torch.where(
        allow_h, h_off, torch.zeros_like(h_off)
      )
      self.is_height_env[env_ids] = allow_h
      # Homie refreshes target_z inside _resample_commands:
      #   z = U(pos_z) * is_height * is_target_height + 0.3
      grasp.resample_target_z(env_ids, is_height=is_height, is_target_height=is_target_height)
      # Homie zeros vel on waist / target, but keeps height sample above.
      return

    # Phase-1 (no grasp): original set_x bands.
    self.is_height_env[env_ids] = is_height
    self.is_standing_env[env_ids] = ~is_vel
    self.vel_command_b[env_ids[~is_vel]] = 0.0
    self.vel_command_w[env_ids[~is_vel]] = 0.0
    r = torch.empty(n, device=self.device)
    h_off = r.uniform_(*self.cfg.ranges.height)
    self.height_command[env_ids] = torch.where(
      is_height, h_off, torch.zeros_like(h_off)
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    base_z = self.robot.data.root_link_pos_w[:, 2]
    target_z = self.cfg.base_height_target + self.height_command
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_height"] += (
      torch.abs(base_z - target_z) / max_command_step
    )

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(env_ids)
    # Idle band: force height_offset=0; keep height-only squat cmds.
    # Viser compute() re-applies the viewed env after this each step.
    idle_ids = (self.is_standing_env & ~self.is_height_env).nonzero(
      as_tuple=False
    ).flatten()
    self.height_command[idle_ids] = 0.0

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Viser joystick: vx/vy/yaw + absolute base height (meters)."""
    del on_change, request_action
    from viser import Icon

    ranges = self.cfg.ranges
    h0 = float(self.cfg.base_height_target)
    # Absolute height slider span from offset range; never collapse to a point.
    h_off_lo, h_off_hi = float(ranges.height[0]), float(ranges.height[1])
    if h_off_hi <= h_off_lo:
      h_off_lo, h_off_hi = -0.5, 0.0
    h_min = h0 + h_off_lo
    h_max = h0 + h_off_hi

    axes = [
      ("lin_vel_x", ranges.lin_vel_x[1]),
      ("lin_vel_y", ranges.lin_vel_y[1]),
      ("ang_vel_z", ranges.ang_vel_z[1]),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      height_abs = server.gui.add_slider(
        "base_height_abs (m)",
        min=h_min,
        max=h_max,
        step=0.01,
        initial_value=h0,
      )
      hint = server.gui.add_number(
        "height_offset (= abs - 0.74)",
        initial_value=0.0,
        step=0.01,
        disabled=True,
      )

      @height_abs.on_update
      def _(_ev, _h=height_abs, _hint=hint, _base=h0) -> None:
        _hint.value = float(_h.value) - _base

      # Live waist pitch readout for the viewed env (updated each compute).
      pitch_rad = server.gui.add_number(
        "waist_pitch (rad)",
        initial_value=0.0,
        step=0.001,
        disabled=True,
        hint="Measured joint angle of viewed env.",
      )
      pitch_deg = server.gui.add_number(
        "waist_pitch (deg)",
        initial_value=0.0,
        step=0.01,
        disabled=True,
      )
      pitch_act = server.gui.add_number(
        "waist_pitch act",
        initial_value=0.0,
        step=0.001,
        disabled=True,
        hint="Policy action for waist_pitch (phase-2 index 13).",
      )

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0
        height_abs.value = h0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._height_abs_slider = height_abs
    self._joystick_get_env_idx = get_env_idx
    self._gui_waist_pitch_rad = pitch_rad
    self._gui_waist_pitch_deg = pitch_deg
    self._gui_waist_pitch_act = pitch_act

  def _update_waist_pitch_gui(self, idx: int) -> None:
    if self._gui_waist_pitch_rad is None or self._waist_pitch_jid is None:
      return
    q = float(self.robot.data.joint_pos[idx, self._waist_pitch_jid].item())
    self._gui_waist_pitch_rad.value = q
    self._gui_waist_pitch_deg.value = q * 57.2957795
    act = 0.0
    try:
      am = self._env.action_manager
      term = am.get_term("joint_pos")
      names = list(getattr(term, "target_names", []) or [])
      if "waist_pitch_joint" in names:
        act = float(am.action[idx, names.index("waist_pitch_joint")].item())
      elif am.total_action_dim >= 14:
        act = float(am.action[idx, 13].item())
    except Exception:  # noqa: BLE001
      act = 0.0
    if self._gui_waist_pitch_act is not None:
      self._gui_waist_pitch_act.value = act

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super(UniformVelocityCommand, self).compute(dt, env_ids)
    if self._joystick_get_env_idx is None:
      return
    idx = self._joystick_get_env_idx()
    self._update_waist_pitch_gui(idx)
    # Height slider always wins when the GUI exists (no need to toggle Enable).
    if self._height_abs_slider is not None:
      abs_h = float(self._height_abs_slider.value)
      h_off = abs_h - float(self.cfg.base_height_target)
      self.height_command[idx] = h_off
      # Keep idle-wipe / standing flags consistent with the slider.
      squatting = abs(h_off) > 1e-3
      self.is_height_env[idx] = squatting
      if squatting:
        self.is_standing_env[idx] = True
    # Velocity: Enable ON → follow sliders; Enable OFF → force zero (stand still).
    # Previously Enable OFF left random train velocities, so the robot kept walking.
    if self._joystick_active():
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value
    elif self._joystick_sliders:
      self.vel_command_b[idx, :] = 0.0


@dataclass(kw_only=True)
class TwistHeightCommandCfg(UniformVelocityCommandCfg):
  """Velocity + height command config."""

  @dataclass
  class Ranges(UniformVelocityCommandCfg.Ranges):
    height: tuple[float, float] = (-0.5, 0.0)

  ranges: Ranges  # type: ignore[assignment]
  base_height_target: float = 0.74

  def build(self, env: ManagerBasedRlEnv) -> TwistHeightCommand:
    return TwistHeightCommand(self, env)
