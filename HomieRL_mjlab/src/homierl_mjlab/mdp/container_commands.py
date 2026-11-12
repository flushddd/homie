"""Container geometry command (10-D task-frame obs, fixed by default).

``o_container = [x_c, y_c, z_b, s_x, s_y, h_c, b_rect, b_circle, t_wall, t_bottom]``

Containers sit upright on a horizontal table (no orientation). Positions are in a
task frame fixed at episode reset: origin = initial pelvis projection on ground,
x forward, y left, z up.

Phase-1/2 use a constant vector (no randomization) matching the design doc example.
Phase-3: grasp targets still sample the full curriculum range. A closed open-top
crate is attached only on hard reaches (``need_waist`` and/or low squat ``body_z``),
with ``presence_prob`` (train default 0.40). Optional play-only relocate can still
move the grasp below/farther than the default hanging EE before placing the crate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from homierl_mjlab.robot.container_spec import (
  BOTTOM_GEOM_NAME,
  CONTAINER_ENTITY_NAME,
  WALL_GEOM_NAMES,
)

if TYPE_CHECKING:
  import viser

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
    # 1 = square bin present (phase-3). Phase-1/2 stay 0 (fixed zeros).
    self.present = torch.zeros(self.num_envs, device=self.device)
    # Task-frame offset of container center relative to grasp (for follow).
    self._grasp_offset_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self._fixed = torch.tensor(cfg.fixed_values, device=self.device, dtype=torch.float32)
    if self._fixed.numel() != 10:
      raise ValueError(f"fixed_values must have length 10, got {self._fixed.numel()}")

    self.metrics["container_rect"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["container_h"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["container_present"] = torch.zeros(self.num_envs, device=self.device)
    # Expand per-world geom fields once so Viser rebuilds wall meshes when
    # sync_walls_to_sim changes size/local pos (closed crate, not baked defaults).
    self._geom_fields_ready = False
    # Viser play: optional Show-container checkbox.
    self._gui_show: Any | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

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

  def _relocate_grasp_below_default_ee(self, env_ids: torch.Tensor) -> None:
    """Move grasp to a band below / farther than the default hanging EE.

    Body-frame convention matches ``GraspTargetCommand``: ``z = U(pos_z) + 0.3``,
    world ``Z = 0.74 + body_z``. Default zero-arm wrists sit roughly near
    standing mid-torso; this band is lower and more forward so the open-top
    crate rim is clearable from above.
    """
    if len(env_ids) == 0:
      return
    grasp = self._env.command_manager.get_term("grasp")
    n = len(env_ids)
    device = self.device
    # Ensure reach mode so EE IK / clearance actually engage.
    grasp.is_waist[env_ids] = 1.0
    grasp.is_target[env_ids] = 1.0
    rx = torch.empty(n, 1, device=device).uniform_(*self.cfg.grasp_pos_x)
    ry = torch.empty(n, 1, device=device).uniform_(*self.cfg.grasp_pos_y)
    rz = torch.empty(n, 1, device=device).uniform_(*self.cfg.grasp_pos_z)
    # Keep existing arm side (y sign) when set; otherwise follow sampled y.
    arm = grasp.target_arm[env_ids].squeeze(-1)
    left = arm > 0.5
    right = arm > 1.5
    if bool(left.any()):
      ry[left] = ry[left].abs()
    if bool(right.any()):
      ry[right] = -ry[right].abs()
    gate = torch.ones(n, 1, device=device)
    grasp.target_position[env_ids, 0:1] = rx * gate
    grasp.target_position[env_ids, 1:2] = ry * gate
    z = rz * gate + 0.3
    grasp.target_position[env_ids, 2:3] = z
    grasp.is_target_height[env_ids] = (torch.abs(z - 0.3) > 1e-3).float() * gate
    grasp._finalize_waist_targets(env_ids, gate)
    grasp.init_robot_poses[env_ids] = self.robot.data.root_link_pos_w[env_ids, :3]
    grasp._freeze_anchor_from_base(env_ids)

  def _hard_reach_mask(self, grasp, env_ids: torch.Tensor) -> torch.Tensor:
    """True where the grasp needs bend/squat help (Homie hard reach)."""
    need_waist = grasp.need_waist[env_ids].squeeze(-1) > 0.5
    body_z = grasp.target_position[env_ids, 2]
    low = body_z < float(self.cfg.low_body_z_max)
    if bool(self.cfg.presence_requires_need_waist) and bool(
      self.cfg.presence_also_on_low_z
    ):
      return need_waist | low
    if bool(self.cfg.presence_requires_need_waist):
      return need_waist
    if bool(self.cfg.presence_also_on_low_z):
      return low
    return torch.ones(len(env_ids), device=self.device, dtype=torch.bool)

  def _resample_around_grasp(self, env_ids: torch.Tensor) -> None:
    """Phase-3: full-range grasp; crate on all hard-reach targets (optional relocate)."""
    n = len(env_ids)
    device = self.device
    grasp = self._env.command_manager.get_term("grasp")
    # Random gate first (play/train can set presence_prob=1 to always try).
    present = (torch.rand(n, device=device) < float(self.cfg.presence_prob)).float()
    if bool(self.cfg.presence_requires_reach):
      is_reach = grasp.is_target[env_ids].squeeze(-1) > 0.5
      present = present * is_reach.float()
    # Only 弯腰 / 下蹲够 targets are eligible; then presence_prob (e.g. 40%).
    if bool(self.cfg.presence_requires_need_waist) or bool(
      self.cfg.presence_also_on_low_z
    ):
      present = present * self._hard_reach_mask(grasp, env_ids).float()
    self.present[env_ids] = present

    present_ids = env_ids[present > 0.5]
    if bool(self.cfg.relocate_grasp_when_present) and len(present_ids) > 0:
      self._relocate_grasp_below_default_ee(present_ids)

    # Grasp target in world → task frame XY; z_b ≈ floor under target.
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
    # Size sampled inside configured ranges (inner cavity / wall / floor).
    inner_x = torch.empty(n, device=device).uniform_(*self.cfg.inner_size_xy)
    inner_y = torch.empty(n, device=device).uniform_(*self.cfg.inner_size_xy)
    h_c = torch.empty(n, device=device).uniform_(*self.cfg.height)
    t_wall = torch.empty(n, device=device).uniform_(*self.cfg.thickness)
    t_bottom = torch.empty(n, device=device).uniform_(*self.cfg.bottom_thickness)
    # Keep a usable cavity: wall/floor cannot eat the whole box.
    min_inner = 2.0 * t_wall + 2.0 * float(self.cfg.target_margin) + 0.04
    inner_x = torch.maximum(inner_x, min_inner)
    inner_y = torch.maximum(inner_y, min_inner)
    h_c = torch.maximum(h_c, t_bottom + 0.05)
    margin = float(self.cfg.target_margin)
    half_in_x = 0.5 * inner_x - margin
    half_in_y = 0.5 * inner_y - margin
    half_in_x = torch.clamp(half_in_x, min=0.02)
    half_in_y = torch.clamp(half_in_y, min=0.02)
    off_x = torch.clamp(off_x, -half_in_x, half_in_x)
    off_y = torch.clamp(off_y, -half_in_y, half_in_y)
    x_c = x_t - off_x
    y_c = y_t - off_y
    # Target near cavity floor; rim sits above so the arm approaches from above.
    z_b = tw[:, 2] - 0.02
    self._grasp_offset_xy[env_ids, 0] = off_x
    self._grasp_offset_xy[env_ids, 1] = off_y

    self.container[env_ids, 0] = x_c * present
    self.container[env_ids, 1] = y_c * present
    self.container[env_ids, 2] = z_b * present
    self.container[env_ids, 3] = inner_x * present
    self.container[env_ids, 4] = inner_y * present
    self.container[env_ids, 5] = h_c * present
    self.container[env_ids, 6] = present  # b_rect
    self.container[env_ids, 7] = 0.0  # b_circle (ellipse later)
    self.container[env_ids, 8] = t_wall * present
    self.container[env_ids, 9] = t_bottom * present

  def _follow_grasp_for_present(self) -> None:
    """Keep present containers centered on the current grasp (play slider / crawl)."""
    ids = (self.present > 0.5).nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
      return
    grasp = self._env.command_manager.get_term("grasp")
    tw = grasp.target_pos_w[ids]
    yaw = self.ref_yaw[ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = tw[:, 0] - self.ref_origin_xy[ids, 0]
    dy = tw[:, 1] - self.ref_origin_xy[ids, 1]
    x_t = c * dx + s * dy
    y_t = -s * dx + c * dy
    off = self._grasp_offset_xy[ids]
    self.container[ids, 0] = x_t - off[:, 0]
    self.container[ids, 1] = y_t - off[:, 1]
    self.container[ids, 2] = tw[:, 2] - 0.02

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

  def _ensure_geom_fields_expanded(self) -> None:
    """Make geom_size/pos per-world so runtime crate layout shows in Viser."""
    if self._geom_fields_ready:
      return
    sim = self._env.sim
    need = ("geom_size", "geom_pos", "geom_rbound", "geom_aabb")
    missing = tuple(f for f in need if f not in sim.expanded_fields)
    if missing:
      sim.expand_model_fields(missing)
    self._geom_fields_ready = True

  def sync_walls_to_sim(self, env_ids: torch.Tensor | None = None) -> None:
    """Push closed-crate params into the mocap ``container_walls`` entity."""
    scene = self._env.scene
    if CONTAINER_ENTITY_NAME not in scene.entities:
      return
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if len(env_ids) == 0:
      return
    self._ensure_geom_fields_expanded()

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

    # Floor plate: outer footprint = inner + 2*t_wall (closed crate bottom).
    t_bottom = torch.clamp(self.container[env_ids, 9], min=0.008)
    bottom_half = torch.zeros(n, 3, device=self.device)
    bottom_pos = torch.zeros(n, 3, device=self.device)
    bottom_half[:, 0] = 0.5 * inner_x + t_wall
    bottom_half[:, 1] = 0.5 * inner_y + t_wall
    bottom_half[:, 2] = 0.5 * t_bottom
    bottom_pos[:, 2] = 0.5 * t_bottom
    bottom_half = torch.where(
      present.unsqueeze(-1), bottom_half, torch.full_like(bottom_half, _EPS_HALF)
    )
    bottom_pos = torch.where(present.unsqueeze(-1), bottom_pos, torch.zeros_like(bottom_pos))

    gnames = list(walls.geom_names)
    try:
      g_order = [gnames.index(n_) for n_ in WALL_GEOM_NAMES]
      bottom_i = gnames.index(BOTTOM_GEOM_NAME)
    except ValueError:
      g_order = list(range(min(4, len(gnames))))
      bottom_i = 4 if len(gnames) > 4 else None
    gids = walls.indexing.geom_ids
    if isinstance(gids, torch.Tensor):
      gid_t = gids[g_order].to(device=self.device)
      gid_bottom = gids[bottom_i].to(device=self.device) if bottom_i is not None else None
    else:
      gid_t = torch.as_tensor(gids, device=self.device)[g_order]
      gid_bottom = (
        torch.as_tensor(gids, device=self.device)[bottom_i]
        if bottom_i is not None
        else None
      )

    env_i = env_ids.to(dtype=torch.long)
    model = self._env.sim.model
    model.geom_pos[env_i[:, None], gid_t[None, :], :] = local_pos
    model.geom_size[env_i[:, None], gid_t[None, :], :] = local_half
    rbound = torch.linalg.norm(local_half, dim=-1)
    model.geom_rbound[env_i[:, None], gid_t[None, :]] = rbound
    model.geom_aabb[env_i[:, None], gid_t[None, :], 1] = local_half
    if gid_bottom is not None:
      model.geom_pos[env_i, gid_bottom, :] = bottom_pos
      model.geom_size[env_i, gid_bottom, :] = bottom_half
      model.geom_rbound[env_i, gid_bottom] = torch.linalg.norm(bottom_half, dim=-1)
      model.geom_aabb[env_i, gid_bottom, 1] = bottom_half

  def _update_metrics(self) -> None:
    self.metrics["container_rect"][:] = self.container[:, 6]
    self.metrics["container_h"][:] = self.container[:, 5]
    self.metrics["container_present"][:] = self.present

  def _gui_wants_container(self) -> bool | None:
    """None = no GUI; else checkbox value."""
    if self._gui_show is None:
      return None
    return bool(self._gui_show.value)

  def _apply_gui_presence(self) -> None:
    """Play: force viewed-env present from the Show-container checkbox."""
    want = self._gui_wants_container()
    if want is None or self._gui_get_env_idx is None:
      return
    idx = int(self._gui_get_env_idx())
    ids = torch.tensor([idx], device=self.device, dtype=torch.long)
    if want:
      self.present[idx] = 1.0
      # First enable / empty geom: place a mid-range crate at the grasp.
      if float(self.container[idx, 5].item()) < 1e-4:
        self._place_fixed_crate_at_grasp(ids)
    else:
      self.present[idx] = 0.0

  def _place_fixed_crate_at_grasp(self, env_ids: torch.Tensor) -> None:
    """Fill play crate geometry around current grasp (no relocate)."""
    if len(env_ids) == 0:
      return
    grasp = self._env.command_manager.get_term("grasp")
    tw = grasp.target_pos_w[env_ids]
    yaw = self.ref_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = tw[:, 0] - self.ref_origin_xy[env_ids, 0]
    dy = tw[:, 1] - self.ref_origin_xy[env_ids, 1]
    x_t = c * dx + s * dy
    y_t = -s * dx + c * dy
    inner_x = 0.5 * (
      float(self.cfg.inner_size_xy[0]) + float(self.cfg.inner_size_xy[1])
    )
    inner_y = inner_x
    h_c = 0.5 * (float(self.cfg.height[0]) + float(self.cfg.height[1]))
    t_wall = 0.5 * (float(self.cfg.thickness[0]) + float(self.cfg.thickness[1]))
    t_bottom = 0.5 * (
      float(self.cfg.bottom_thickness[0]) + float(self.cfg.bottom_thickness[1])
    )
    self._grasp_offset_xy[env_ids] = 0.0
    self.container[env_ids, 0] = x_t
    self.container[env_ids, 1] = y_t
    self.container[env_ids, 2] = tw[:, 2] - 0.02
    self.container[env_ids, 3] = inner_x
    self.container[env_ids, 4] = inner_y
    self.container[env_ids, 5] = h_c
    self.container[env_ids, 6] = 1.0
    self.container[env_ids, 7] = 0.0
    self.container[env_ids, 8] = t_wall
    self.container[env_ids, 9] = t_bottom
    self.present[env_ids] = 1.0

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Viser play: toggle whether the closed crate is present on the viewed env."""
    del on_change, request_action
    if not bool(self.cfg.sample_around_grasp):
      return
    if CONTAINER_ENTITY_NAME not in self._env.scene.entities:
      return
    # Default OFF: checkbox forces presence every step when ON; starting True
    # drops a crate on the grasp and IK/COM fight the walls.
    with server.gui.add_folder(name.capitalize()):
      show = server.gui.add_checkbox(
        "Show container",
        initial_value=False,
        hint="ON=closed crate around grasp target. OFF=hide crate (no SDF walls).",
      )
    self._gui_show = show
    self._gui_get_env_idx = get_env_idx

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    # Re-sync every step: reset_scene_to_default re-buries mocap walls on reset.
    # Also re-center on grasp so play Viser / arm crawl keep a closed bin.
    del env_ids
    self._apply_gui_presence()
    if self.cfg.sample_around_grasp and bool(self.present.any()):
      self._follow_grasp_for_present()
      self.sync_walls_to_sim()
    elif self._gui_show is not None:
      # Checkbox may have just turned OFF — bury walls for the viewed env.
      self.sync_walls_to_sim()
    elif bool(self.present.any()):
      self.sync_walls_to_sim()


@dataclass(kw_only=True)
class ContainerGeometryCommandCfg(CommandTermCfg):
  """Fixed (default) or optional random upright container geometry."""

  entity_name: str = "robot"
  # Rarely resample; values are constant when randomize=False.
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  debug_vis: bool = False
  randomize: bool = False
  fixed_values: tuple[float, ...] = field(default_factory=lambda: _FIXED_CONTAINER)
  # Phase-3: sample square bin around grasp (overrides randomize).
  # Phase-1/2 leave this False and keep ``fixed_values`` zeros.
  sample_around_grasp: bool = False
  # Among eligible hard-reach targets, this fraction get a bin (train: 0.40).
  presence_prob: float = 0.40
  # Train: only put crates on reach envs. Play can set False to always show one.
  presence_requires_reach: bool = True
  # Homie ``need_waist`` (beyond arm reach → bend) and/or low squat body_z.
  presence_requires_need_waist: bool = True
  presence_also_on_low_z: bool = True
  # ``target_position[:,2]`` (already ``U(pos_z)+0.3``); world_z = 0.74 + body_z.
  low_body_z_max: float = 0.15
  # Play/debug only: move grasp below/farther than default hanging EE first.
  # Train keeps full grasp curriculum — leave False.
  relocate_grasp_when_present: bool = False
  grasp_pos_x: tuple[float, float] = (0.38, 0.55)
  grasp_pos_y: tuple[float, float] = (-0.22, 0.22)
  grasp_pos_z: tuple[float, float] = (-0.55, -0.38)
  # Closed open-top crate size ranges (meters). ``uniform_(lo, hi)`` per env.
  # Desktop-bin scale: ~30–42 cm cavity, ~10–16 cm tall (clearable rim).
  inner_size_xy: tuple[float, float] = (0.30, 0.42)
  height: tuple[float, float] = (0.10, 0.16)
  thickness: tuple[float, float] = (0.020, 0.030)
  bottom_thickness: tuple[float, float] = (0.012, 0.020)
  target_offset_xy: tuple[float, float] = (-0.04, 0.04)
  target_margin: float = 0.05
  # Used only if randomize=True (legacy free-place path).
  pos_x: tuple[float, float] = (0.40, 0.80)
  pos_y: tuple[float, float] = (-0.30, 0.30)
  z_b: tuple[float, float] = (0.70, 0.80)
  size_xy: tuple[float, float] = (0.20, 0.50)
  circle_prob: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> ContainerGeometryCommand:
    return ContainerGeometryCommand(self, env)
