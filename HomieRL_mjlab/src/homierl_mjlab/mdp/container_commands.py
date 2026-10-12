"""Container geometry command (10-D task-frame obs, fixed by default).

``o_container = [x_c, y_c, z_b, s_x, s_y, h_c, b_rect, b_circle, t_wall, t_bottom]``

Containers sit upright on a horizontal table (no orientation). Positions are in a
task frame fixed at episode reset: origin = initial pelvis projection on ground,
x forward, y left, z up.

Phase-1/2 use a constant vector (no randomization) matching the design doc example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Phase-1/2: placeholder zeros (no container signal yet).
_FIXED_CONTAINER = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class ContainerGeometryCommand(CommandTerm):
  """Fixed upright container geometry in a task frame (optional randomize)."""

  cfg: ContainerGeometryCommandCfg

  def __init__(self, cfg: ContainerGeometryCommandCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    self.ref_origin_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.ref_yaw = torch.zeros(self.num_envs, device=self.device)
    self.container = torch.zeros(self.num_envs, 10, device=self.device)
    self._fixed = torch.tensor(cfg.fixed_values, device=self.device, dtype=torch.float32)
    if self._fixed.numel() != 10:
      raise ValueError(f"fixed_values must have length 10, got {self._fixed.numel()}")

    self.metrics["container_rect"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["container_h"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.container

  def _capture_task_frame(self, env_ids: torch.Tensor) -> None:
    root = self.robot.data.root_link_pos_w[env_ids, :3]
    self.ref_origin_xy[env_ids] = root[:, :2]
    quat = self.robot.data.root_link_quat_w[env_ids]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    self.ref_yaw[env_ids] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

  def _fill_fixed(self, env_ids: torch.Tensor) -> None:
    self.container[env_ids] = self._fixed

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self._capture_task_frame(env_ids)
    if not self.cfg.randomize:
      self._fill_fixed(env_ids)
      return

    # Optional randomization path (disabled for phase-1/2 by default).
    n = len(env_ids)
    device = self.device
    x_c = torch.empty(n, device=device).uniform_(*self.cfg.pos_x)
    y_c = torch.empty(n, device=device).uniform_(*self.cfg.pos_y)
    z_b = torch.empty(n, device=device).uniform_(*self.cfg.z_b)
    s_x = torch.empty(n, device=device).uniform_(*self.cfg.size_xy)
    s_y = torch.empty(n, device=device).uniform_(*self.cfg.size_xy)
    h_c = torch.empty(n, device=device).uniform_(*self.cfg.height)
    t_wall = torch.empty(n, device=device).uniform_(*self.cfg.thickness)
    t_bottom = torch.empty(n, device=device).uniform_(*self.cfg.thickness)
    is_circle = torch.rand(n, device=device) < float(self.cfg.circle_prob)
    diam = 0.5 * (s_x + s_y)
    s_x = torch.where(is_circle, diam, s_x)
    s_y = torch.where(is_circle, diam, s_y)
    min_xy = 2.0 * t_wall + 1e-3
    s_x = torch.maximum(s_x, min_xy)
    s_y = torch.maximum(s_y, min_xy)
    h_c = torch.maximum(h_c, t_bottom + 1e-3)
    self.container[env_ids, 0] = x_c
    self.container[env_ids, 1] = y_c
    self.container[env_ids, 2] = z_b
    self.container[env_ids, 3] = s_x
    self.container[env_ids, 4] = s_y
    self.container[env_ids, 5] = h_c
    self.container[env_ids, 6] = (~is_circle).float()
    self.container[env_ids, 7] = is_circle.float()
    self.container[env_ids, 8] = t_wall
    self.container[env_ids, 9] = t_bottom

  def container_pos_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    xy = self.container[env_ids, 0:2]
    yaw = self.ref_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_w = c * xy[:, 0] - s * xy[:, 1] + self.ref_origin_xy[env_ids, 0]
    y_w = s * xy[:, 0] + c * xy[:, 1] + self.ref_origin_xy[env_ids, 1]
    z_w = self.container[env_ids, 2]
    return torch.stack((x_w, y_w, z_w), dim=-1)

  def _update_metrics(self) -> None:
    self.metrics["container_rect"][:] = self.container[:, 6]
    self.metrics["container_h"][:] = self.container[:, 5]

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids


@dataclass(kw_only=True)
class ContainerGeometryCommandCfg(CommandTermCfg):
  """Fixed (default) or optional random upright container geometry."""

  entity_name: str = "robot"
  # Rarely resample; values are constant when randomize=False.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False
  randomize: bool = False
  fixed_values: tuple[float, ...] = field(default_factory=lambda: _FIXED_CONTAINER)
  # Used only if randomize=True
  pos_x: tuple[float, float] = (0.40, 0.80)
  pos_y: tuple[float, float] = (-0.30, 0.30)
  z_b: tuple[float, float] = (0.70, 0.80)
  size_xy: tuple[float, float] = (0.20, 0.50)
  height: tuple[float, float] = (0.10, 0.30)
  thickness: tuple[float, float] = (0.005, 0.020)
  circle_prob: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> ContainerGeometryCommand:
    return ContainerGeometryCommand(self, env)
