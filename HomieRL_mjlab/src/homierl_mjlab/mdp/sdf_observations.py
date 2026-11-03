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
  # Palm TCP (Inspire palm sensor offset on wrist_yaw), not wrist origin.
  ("left_wrist_yaw_link", (0.1611, -0.012296, 0.00036819), 0.05),
  ("right_shoulder_pitch_link", (0.0, -0.05, 0.0), 0.06),
  ("right_elbow_link", (0.0, 0.0, 0.0), 0.055),
  ("right_wrist_yaw_link", (0.1611, 0.012296, 0.00036819), 0.05),
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
  from mjlab.utils.lab_api.math import quat_apply

  centers = []
  radii = []
  for link, offset, radius in _ARM_SPHERES:
    if link not in bnames:
      continue
    bi = bnames.index(link)
    pos = robot.data.body_link_pos_w[:, bi]
    quat = robot.data.body_link_quat_w[:, bi]
    off = torch.tensor(offset, device=env.device, dtype=pos.dtype)
    centers.append(pos + quat_apply(quat, off.expand(pos.shape[0], 3)))
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
  """Base-frame palm→anchor error for active arm (3-D), zeros if no arm."""
  from homierl_mjlab.mdp.palm import link_point_pos_w, palm_offset_tensor

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
  left_off = palm_offset_tensor(env.device, torch.float32, left=True)
  right_off = palm_offset_tensor(env.device, torch.float32, left=False)
  left_w = link_point_pos_w(robot, left_i, left_off)
  right_w = link_point_pos_w(robot, right_i, right_off)
  ee_w = torch.where((arm < 1.5).unsqueeze(-1), left_w, right_w)
  ee_b = world_to_base(robot, ee_w)
  tgt_b = world_to_base(robot, grasp.target_pos_w)
  err = tgt_b - ee_b
  active = (arm > 0.5).float().unsqueeze(-1)
  return err * active


def container_ik_avoidance_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  ee_w: torch.Tensor,
  target_w: torch.Tensor,
  *,
  command_name: str = "container",
  d_safe: float = 0.05,
  lift_margin: float = 0.05,
  xy_slow: float = 0.25,
  repulsion_gain: float = 1.0,
  d_max: float = 0.40,
  fd_eps: float = 0.01,
) -> torch.Tensor:
  """Base-frame bias added to IK Cartesian error for open-top crate avoidance.

  1. **Lift-over**: while EE is below the rim, chase a waypoint above the rim
     and slow XY motion so the arm does not plow through the wall.
  2. **SDF repulsion**: finite-difference grad of wall distance at the EE;
     push out when ``phi < d_safe``.
  """
  from homierl_mjlab.mdp.base_frame import rotate_world_to_base

  m = int(env_ids.numel())
  device = ee_w.device
  dtype = ee_w.dtype
  zero = torch.zeros(m, 3, device=device, dtype=dtype)
  if m == 0:
    return zero

  cont = env.command_manager.get_term(command_name)
  present = cont.present[env_ids] > 0.5
  if not bool(present.any()):
    return zero

  robot = env.scene["robot"]
  cx, cy, z_b, inner_x, inner_y, h_c, t_wall = _container_world_params(cont)
  cx_i = cx[env_ids]
  cy_i = cy[env_ids]
  z_b_i = z_b[env_ids]
  inner_x_i = inner_x[env_ids]
  inner_y_i = inner_y[env_ids]
  h_c_i = h_c[env_ids]
  t_wall_i = t_wall[env_ids]
  rim_z = z_b_i + h_c_i
  half_in_x = 0.5 * inner_x_i
  half_in_y = 0.5 * inner_y_i

  # --- Lift-over waypoint (world) ---
  wp = target_w.clone()
  below = ee_w[:, 2] < (rim_z + float(lift_margin))
  # Outside inner cavity in XY (still approaching / skirting walls).
  outside = (torch.abs(ee_w[:, 0] - cx_i) > half_in_x) | (
    torch.abs(ee_w[:, 1] - cy_i) > half_in_y
  )
  lift = present & below
  skirt = lift & outside
  # Raise waypoint above rim until EE clears.
  wp[:, 2] = torch.where(
    lift, torch.maximum(wp[:, 2], rim_z + float(lift_margin)), wp[:, 2]
  )
  # While skirting below the rim, only creep in XY toward the target.
  creep = float(xy_slow)
  wp_xy = ee_w[:, :2] + creep * (target_w[:, :2] - ee_w[:, :2])
  wp[:, 0] = torch.where(skirt, wp_xy[:, 0], wp[:, 0])
  wp[:, 1] = torch.where(skirt, wp_xy[:, 1], wp[:, 1])

  wp_b = world_to_base(robot, wp, env_ids)
  ee_b = world_to_base(robot, ee_w, env_ids)
  tgt_b = world_to_base(robot, target_w, env_ids)
  # Bias = (waypoint - ee) - (target - ee) = waypoint - target in base.
  lift_bias = (wp_b - tgt_b) * present.unsqueeze(-1).to(dtype=dtype)

  # --- SDF repulsion at EE ---
  pts = ee_w.unsqueeze(1)  # (M,1,3)
  phi, _ = query_open_box_distance(
    pts,
    present=cont.present[env_ids],
    cx=cx_i,
    cy=cy_i,
    z_b=z_b_i,
    inner_x=inner_x_i,
    inner_y=inner_y_i,
    h_c=h_c_i,
    t_wall=t_wall_i,
    d_max=d_max,
  )
  phi0 = phi[:, 0]
  eps = float(fd_eps)
  eye = torch.eye(3, device=device, dtype=dtype)
  grads = []
  for ax in range(3):
    pts_d = pts + eps * eye[ax].view(1, 1, 3)
    phi_d, _ = query_open_box_distance(
      pts_d,
      present=cont.present[env_ids],
      cx=cx_i,
      cy=cy_i,
      z_b=z_b_i,
      inner_x=inner_x_i,
      inner_y=inner_y_i,
      h_c=h_c_i,
      t_wall=t_wall_i,
      d_max=d_max,
    )
    grads.append((phi_d[:, 0] - phi0) / eps)
  grad_w = torch.stack(grads, dim=-1)
  grad_n = torch.linalg.norm(grad_w, dim=-1, keepdim=True).clamp(min=1e-6)
  grad_hat_w = grad_w / grad_n
  viol = torch.clamp(float(d_safe) - phi0, min=0.0)
  rep_w = (float(repulsion_gain) * viol).unsqueeze(-1) * grad_hat_w
  rep_b = rotate_world_to_base(robot, rep_w, env_ids)
  rep_b = rep_b * present.unsqueeze(-1).to(dtype=dtype)

  return lift_bias + rep_b
