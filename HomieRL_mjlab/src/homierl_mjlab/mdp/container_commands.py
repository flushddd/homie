"""Container geometry command (10-D task-frame obs, fixed by default).

``o_container = [x_c, y_c, z_b, s_x, s_y, h_c, b_rect, b_circle, t_wall, t_bottom]``

Containers sit upright on a horizontal table (no orientation). Positions are in a
task frame fixed at episode reset: origin = initial pelvis projection on ground,
x forward, y left, z up.

Phase-1/2 use a constant vector (no randomization) matching the design doc example.
Phase-3 samples open-bottom square walls around the grasp target and syncs a
mocap MuJoCo entity (``container_walls``) for collide + viz; analytic SDF uses
the same params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from homierl_mjlab.robot.container_spec import CONTAINER_ENTITY_NAME, WALL_GEOM_NAMES

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

# Phase-1/2: placeholder zeros (no container signal yet).
_FIXED_CONTAINER = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# Hide absent containers underground with near-zero size.
_HIDE_Z = -5.0
_EPS_HALF = 1e-3


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
    # 1 = open-bottom square walls present (phase-3).
    self.present = torch.zeros(self.num_envs, device=self.device)
    self._fixed = torch.tensor(cfg.fixed_values, device=self.device, dtype=torch.float32)
    if self._fixed.numel() != 10:
      raise ValueError(f"fixed_values must have length 10, got {self._fixed.numel()}")

    self.metrics["container_rect"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["container_h"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["container_present"] = torch.zeros(self.num_envs, device=self.device)

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
    if self.cfg.sample_around_grasp:
      self._resample_around_grasp(env_ids)
      self.sync_walls_to_sim(env_ids)
      return
    if not self.cfg.randomize:
      self._fill_fixed(env_ids)
      self.present[env_ids] = 0.0
      self.sync_walls_to_sim(env_ids)
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
    self.present[env_ids] = 1.0
    self.sync_walls_to_sim(env_ids)

  def _resample_around_grasp(self, env_ids: torch.Tensor) -> None:
    """Phase-3: optional open-bottom square walls around grasp target (world→task)."""
    n = len(env_ids)
    device = self.device
    present = (torch.rand(n, device=device) < float(self.cfg.presence_prob)).float()
    self.present[env_ids] = present

    # Grasp target in world → task frame XY; z_b ≈ table under target.
    grasp = self._env.command_manager.get_term("grasp")
    tw = grasp.target_pos_w[env_ids]
    yaw = self.ref_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = tw[:, 0] - self.ref_origin_xy[env_ids, 0]
    dy = tw[:, 1] - self.ref_origin_xy[env_ids, 1]
    x_t = c * dx + s * dy
    y_t = -s * dx + c * dy

    # Container center near target with small offset; keep target inside with margin.
    off_x = torch.empty(n, device=device).uniform_(*self.cfg.target_offset_xy)
    off_y = torch.empty(n, device=device).uniform_(*self.cfg.target_offset_xy)
    inner_x = torch.empty(n, device=device).uniform_(*self.cfg.inner_size_xy)
    inner_y = torch.empty(n, device=device).uniform_(*self.cfg.inner_size_xy)
    h_c = torch.empty(n, device=device).uniform_(*self.cfg.height)
    t_wall = torch.empty(n, device=device).uniform_(*self.cfg.thickness)
    margin = float(self.cfg.target_margin)
    # Ensure target stays inside cavity with margin.
    half_in_x = 0.5 * inner_x - margin
    half_in_y = 0.5 * inner_y - margin
    half_in_x = torch.clamp(half_in_x, min=0.02)
    half_in_y = torch.clamp(half_in_y, min=0.02)
    off_x = torch.clamp(off_x, -half_in_x, half_in_x)
    off_y = torch.clamp(off_y, -half_in_y, half_in_y)
    x_c = x_t - off_x
    y_c = y_t - off_y
    z_b = tw[:, 2] - 0.02  # slightly below target (desktop support)

    self.container[env_ids, 0] = x_c * present
    self.container[env_ids, 1] = y_c * present
    self.container[env_ids, 2] = z_b * present
    self.container[env_ids, 3] = inner_x * present
    self.container[env_ids, 4] = inner_y * present
    self.container[env_ids, 5] = h_c * present
    self.container[env_ids, 6] = present  # b_rect
    self.container[env_ids, 7] = 0.0  # b_circle (ellipse later)
    self.container[env_ids, 8] = t_wall * present
    self.container[env_ids, 9] = 0.0  # open bottom

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

  def sync_walls_to_sim(self, env_ids: torch.Tensor | None = None) -> None:
    """Push open-box params into the mocap ``container_walls`` entity (if present)."""
    scene = self._env.scene
    if CONTAINER_ENTITY_NAME not in scene.entities:
      return
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return

    walls: Entity = scene[CONTAINER_ENTITY_NAME]
    n = len(env_ids)
    present = self.present[env_ids] > 0.5
    pos_w = self.container_pos_w(env_ids)  # bottom-center in world (no env origin)
    yaw = self.ref_yaw[env_ids]
    half_yaw = 0.5 * yaw
    quat = torch.stack(
      (
        torch.cos(half_yaw),
        torch.zeros(n, device=self.device),
        torch.zeros(n, device=self.device),
        torch.sin(half_yaw),
      ),
      dim=-1,
    )
    # Absent → bury underground.
    pos = pos_w.clone()
    pos[:, 2] = torch.where(present, pos[:, 2], torch.full_like(pos[:, 2], _HIDE_Z))
    pos = pos + scene.env_origins[env_ids]
    pose = torch.cat((pos, quat), dim=-1)
    walls.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    # Local wall layout (same as ``open_box_wall_centers_half`` relative to z_b).
    inner_x = self.container[env_ids, 3]
    inner_y = self.container[env_ids, 4]
    h_c = self.container[env_ids, 5]
    t_wall = self.container[env_ids, 8]
    hx = 0.5 * inner_x + t_wall
    hy = 0.5 * inner_y + t_wall
    half_h = 0.5 * h_c
    half_t = 0.5 * t_wall
    z_mid = half_h

    # (N, 4, 3) local pos / half-size — order wall_xp, xn, yp, yn.
    local_pos = torch.zeros(n, 4, 3, device=self.device)
    local_half = torch.zeros(n, 4, 3, device=self.device)
    local_pos[:, 0, 0] = 0.5 * inner_x + half_t
    local_pos[:, 0, 2] = z_mid
    local_half[:, 0, 0] = half_t
    local_half[:, 0, 1] = hy
    local_half[:, 0, 2] = half_h
    local_pos[:, 1, 0] = -(0.5 * inner_x + half_t)
    local_pos[:, 1, 2] = z_mid
    local_half[:, 1, 0] = half_t
    local_half[:, 1, 1] = hy
    local_half[:, 1, 2] = half_h
    local_pos[:, 2, 1] = 0.5 * inner_y + half_t
    local_pos[:, 2, 2] = z_mid
    local_half[:, 2, 0] = hx
    local_half[:, 2, 1] = half_t
    local_half[:, 2, 2] = half_h
    local_pos[:, 3, 1] = -(0.5 * inner_y + half_t)
    local_pos[:, 3, 2] = z_mid
    local_half[:, 3, 0] = hx
    local_half[:, 3, 1] = half_t
    local_half[:, 3, 2] = half_h

    # Collapse absent envs to epsilon boxes at origin (already buried via mocap).
    local_half = torch.where(
      present.view(n, 1, 1),
      local_half,
      torch.full_like(local_half, _EPS_HALF),
    )
    local_pos = torch.where(present.view(n, 1, 1), local_pos, torch.zeros_like(local_pos))

    gnames = list(walls.geom_names)
    try:
      g_order = [gnames.index(n_) for n_ in WALL_GEOM_NAMES]
    except ValueError:
      g_order = list(range(min(4, len(gnames))))
    gids = walls.indexing.geom_ids
    if isinstance(gids, torch.Tensor):
      gid_t = gids[g_order].to(device=self.device)
    else:
      gid_t = torch.as_tensor(gids, device=self.device)[g_order]

    env_i = env_ids.to(dtype=torch.long)
    model = self._env.sim.model
    model.geom_pos[env_i[:, None], gid_t[None, :], :] = local_pos
    model.geom_size[env_i[:, None], gid_t[None, :], :] = local_half
    # Broadphase bounds for boxes.
    rbound = torch.linalg.norm(local_half, dim=-1)
    model.geom_rbound[env_i[:, None], gid_t[None, :]] = rbound
    model.geom_aabb[env_i[:, None], gid_t[None, :], 1] = local_half

  def _update_metrics(self) -> None:
    self.metrics["container_rect"][:] = self.container[:, 6]
    self.metrics["container_h"][:] = self.container[:, 5]
    self.metrics["container_present"][:] = self.present

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
  # Phase-3: sample open-bottom square walls around grasp (overrides randomize).
  sample_around_grasp: bool = False
  presence_prob: float = 0.75
  inner_size_xy: tuple[float, float] = (0.28, 0.45)
  target_offset_xy: tuple[float, float] = (-0.06, 0.06)
  target_margin: float = 0.06
  # Used only if randomize=True
  pos_x: tuple[float, float] = (0.40, 0.80)
  pos_y: tuple[float, float] = (-0.30, 0.30)
  z_b: tuple[float, float] = (0.70, 0.80)
  size_xy: tuple[float, float] = (0.20, 0.50)
  height: tuple[float, float] = (0.08, 0.18)
  thickness: tuple[float, float] = (0.01, 0.025)
  circle_prob: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> ContainerGeometryCommand:
    return ContainerGeometryCommand(self, env)
