"""Phase-3 geometry observations: base-frame SDF queries + arm clearance.

Query grids and EE error are expressed in Homie **base** (pelvis + yaw) so the
policy interface matches deploy (point-cloud SDF in base). Analytic open-box
distance is still evaluated in world after transforming query points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from homierl_mjlab.mdp.base_frame import world_to_base
from homierl_mjlab.mdp.sdf_geometry import (
  make_shoulder_query_grid,
  query_open_box_distance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Cached query grids keyed by (device, count) — offsets in base frame.
_QUERY_CACHE: dict[tuple[str, int], torch.Tensor] = {}

_ARM_SPHERES = (
  ("left_shoulder_pitch_link", (0.0, 0.05, 0.0), 0.06),
  ("left_elbow_link", (0.0, 0.0, 0.0), 0.055),
  ("left_wrist_yaw_link", (0.0, 0.0, 0.0), 0.05),
  ("right_shoulder_pitch_link", (0.0, -0.05, 0.0), 0.06),
  ("right_elbow_link", (0.0, 0.0, 0.0), 0.055),
  ("right_wrist_yaw_link", (0.0, 0.0, 0.0), 0.05),
)


def _query_grid(device: torch.device, count: int) -> torch.Tensor:
  """Fixed sample points in base frame (forward / left / up from pelvis)."""
  key = (str(device), int(count))
  if key not in _QUERY_CACHE:
    # Same spatial extents as before, but origin = pelvis (not shoulder).
    _QUERY_CACHE[key] = make_shoulder_query_grid(
      count=count,
      x_range=(0.10, 0.70),
      y_range=(-0.45, 0.45),
      z_range=(-0.25, 0.35),
      device=device,
    )
  return _QUERY_CACHE[key]


def _base_frame_points_w(
  env: ManagerBasedRlEnv,
  points_b: torch.Tensor,
) -> torch.Tensor:
  """Map base-frame query grid ``(K,3)`` → world ``(N,K,3)``."""
  from mjlab.utils.lab_api.math import quat_apply

  from homierl_mjlab.mdp.base_frame import heading_quat, pelvis_pos

  robot = env.scene["robot"]
  n = env.num_envs
  k = points_b.shape[0]
  origin = pelvis_pos(robot)  # (N,3)
  heading = heading_quat(robot)  # (N,4)
  px = points_b[:, 0].view(1, k).expand(n, k)
  py = points_b[:, 1].view(1, k).expand(n, k)
  pz = points_b[:, 2].view(1, k).expand(n, k)
  vec = torch.stack((px, py, pz), dim=-1).reshape(n * k, 3)
  h = heading.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
  rotated = quat_apply(h, vec).view(n, k, 3)
  return origin.unsqueeze(1) + rotated


def _container_world_params(cont) -> tuple[torch.Tensor, ...]:
  yaw = cont.ref_yaw
  c, s = torch.cos(yaw), torch.sin(yaw)
  x_t = cont.container[:, 0]
  y_t = cont.container[:, 1]
  cx = c * x_t - s * y_t + cont.ref_origin_xy[:, 0]
  cy = s * x_t + c * y_t + cont.ref_origin_xy[:, 1]
  return (
    cx,
    cy,
    cont.container[:, 2],
    cont.container[:, 3],
    cont.container[:, 4],
    cont.container[:, 5],
    cont.container[:, 8],
  )


def sdf_query_field(
  env: ManagerBasedRlEnv,
  command_name: str = "container",
  query_count: int = 64,
  d_max: float = 0.40,
  include_valid: bool = True,
) -> torch.Tensor:
  """Normalized SDF at fixed **base-frame** queries: ``s=clip(phi)/d_max`` (+ valid)."""
  cont = env.command_manager.get_term(command_name)
  grid = _query_grid(env.device, query_count)
  pts_w = _base_frame_points_w(env, grid)
  cx, cy, z_b, inner_x, inner_y, h_c, t_wall = _container_world_params(cont)
  phi, valid = query_open_box_distance(
    pts_w,
    present=cont.present,
    cx=cx,
    cy=cy,
    z_b=z_b,
    inner_x=inner_x,
    inner_y=inner_y,
    h_c=h_c,
    t_wall=t_wall,
    d_max=d_max,
  )
  s_norm = phi / float(d_max)
  if include_valid:
    return torch.cat((s_norm, valid), dim=-1)
  return s_norm


def arm_clearance_field(
  env: ManagerBasedRlEnv,
  command_name: str = "container",
  d_max: float = 0.40,
) -> torch.Tensor:
  """Net clearance ``c_j = phi(p_j) - r_j`` normalized by ``d_max`` for arm spheres."""
  robot = env.scene["robot"]
  cont = env.command_manager.get_term(command_name)
  bnames = list(robot.body_names)
  centers = []
  radii = []
  for link, offset, radius in _ARM_SPHERES:
    if link not in bnames:
      continue
    bi = bnames.index(link)
    pos = robot.data.body_link_pos_w[:, bi]
    off = torch.tensor(offset, device=env.device, dtype=pos.dtype)
    centers.append(pos + off.unsqueeze(0))
    radii.append(radius)
  if not centers:
    return torch.zeros(env.num_envs, 1, device=env.device)
  pts = torch.stack(centers, dim=1)  # (N, M, 3) world — distance invariant
  cx, cy, z_b, inner_x, inner_y, h_c, t_wall = _container_world_params(cont)
  phi, _ = query_open_box_distance(
    pts,
    present=cont.present,
    cx=cx,
    cy=cy,
    z_b=z_b,
    inner_x=inner_x,
    inner_y=inner_y,
    h_c=h_c,
    t_wall=t_wall,
    d_max=d_max,
  )
  r = torch.tensor(radii, device=env.device, dtype=phi.dtype).view(1, -1)
  clearance = (phi - r) / float(d_max)
  return clearance


def ee_target_error(
  env: ManagerBasedRlEnv,
  grasp_command_name: str = "grasp",
) -> torch.Tensor:
  """Base-frame EE→anchor error for active arm (3-D), zeros if no arm."""
  robot = env.scene["robot"]
  grasp = env.command_manager.get_term(grasp_command_name)
  bnames = list(robot.body_names)
  left_i = (
    bnames.index("left_wrist_yaw_link")
    if "left_wrist_yaw_link" in bnames
    else bnames.index("left_elbow_link")
  )
  right_i = (
    bnames.index("right_wrist_yaw_link")
    if "right_wrist_yaw_link" in bnames
    else bnames.index("right_elbow_link")
  )
  arm = grasp.target_arm.squeeze(-1)
  ee_w = torch.where(
    (arm < 1.5).unsqueeze(-1),
    robot.data.body_link_pos_w[:, left_i],
    robot.data.body_link_pos_w[:, right_i],
  )
  ee_b = world_to_base(robot, ee_w)
  tgt_b = world_to_base(robot, grasp.target_pos_w)
  err = tgt_b - ee_b
  active = (arm > 0.5).float().unsqueeze(-1)
  return err * active
