"""Phase-3 geometry observations: fixed shoulder-frame SDF queries + arm clearance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from homierl_mjlab.mdp.sdf_geometry import (
  make_shoulder_query_grid,
  query_open_box_distance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Cached query grids keyed by (device, count).
_QUERY_CACHE: dict[tuple[str, int], torch.Tensor] = {}

# Collision-sphere proxies (link name, local offset, radius). Tunable.
_ARM_SPHERES = (
  ("left_shoulder_pitch_link", (0.0, 0.05, 0.0), 0.06),
  ("left_elbow_link", (0.0, 0.0, 0.0), 0.055),
  ("left_wrist_yaw_link", (0.0, 0.0, 0.0), 0.05),
  ("right_shoulder_pitch_link", (0.0, -0.05, 0.0), 0.06),
  ("right_elbow_link", (0.0, 0.0, 0.0), 0.055),
  ("right_wrist_yaw_link", (0.0, 0.0, 0.0), 0.05),
)


def _query_grid(device: torch.device, count: int) -> torch.Tensor:
  key = (str(device), int(count))
  if key not in _QUERY_CACHE:
    _QUERY_CACHE[key] = make_shoulder_query_grid(
      count=count,
      x_range=(0.10, 0.70),
      y_range=(-0.45, 0.45),
      z_range=(-0.25, 0.35),
      device=device,
    )
  return _QUERY_CACHE[key]


def _shoulder_frame_points_w(
  env: ManagerBasedRlEnv,
  points_s: torch.Tensor,
) -> torch.Tensor:
  """Map shoulder-frame points to world using torso yaw heading + left/right shoulder."""
  robot = env.scene["robot"]
  grasp = env.command_manager.get_term("grasp")
  bnames = list(robot.body_names)
  left_i = bnames.index("left_shoulder_pitch_link")
  right_i = bnames.index("right_shoulder_pitch_link")
  arm = grasp.target_arm.squeeze(-1)
  use_left = arm < 1.5
  sh = torch.where(
    use_left.unsqueeze(-1),
    robot.data.body_link_pos_w[:, left_i],
    robot.data.body_link_pos_w[:, right_i],
  )
  # Heading from root yaw (ignore roll/pitch for query frame; waist bend later).
  quat = robot.data.root_link_quat_w
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  c, s = torch.cos(yaw), torch.sin(yaw)
  # R_WS * p_S + t
  px, py, pz = points_s[:, 0], points_s[:, 1], points_s[:, 2]
  # Broadcast: points_s is (K,3), need (N,K,3)
  n = sh.shape[0]
  k = points_s.shape[0]
  px = px.view(1, k).expand(n, k)
  py = py.view(1, k).expand(n, k)
  pz = pz.view(1, k).expand(n, k)
  c = c.view(n, 1)
  s = s.view(n, 1)
  xw = c * px - s * py + sh[:, 0:1]
  yw = s * px + c * py + sh[:, 1:2]
  zw = pz + sh[:, 2:3]
  return torch.stack((xw, yw, zw), dim=-1)


def sdf_query_field(
  env: ManagerBasedRlEnv,
  command_name: str = "container",
  query_count: int = 64,
  d_max: float = 0.40,
  include_valid: bool = True,
) -> torch.Tensor:
  """Normalized SDF at fixed shoulder-frame queries: ``s=clip(phi)/d_max`` (+ valid)."""
  cont = env.command_manager.get_term(command_name)
  grid = _query_grid(env.device, query_count)
  pts_w = _shoulder_frame_points_w(env, grid)
  # Container params in task frame → world XY via container helpers.
  # Use world centers: convert task (x_c,y_c) with ref frame.
  yaw = cont.ref_yaw
  c, s = torch.cos(yaw), torch.sin(yaw)
  x_t = cont.container[:, 0]
  y_t = cont.container[:, 1]
  cx = c * x_t - s * y_t + cont.ref_origin_xy[:, 0]
  cy = s * x_t + c * y_t + cont.ref_origin_xy[:, 1]
  z_b = cont.container[:, 2]
  inner_x = cont.container[:, 3]
  inner_y = cont.container[:, 4]
  h_c = cont.container[:, 5]
  t_wall = cont.container[:, 8]
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
  pts = torch.stack(centers, dim=1)  # (N, M, 3)
  yaw = cont.ref_yaw
  c, s = torch.cos(yaw), torch.sin(yaw)
  x_t = cont.container[:, 0]
  y_t = cont.container[:, 1]
  cx = c * x_t - s * y_t + cont.ref_origin_xy[:, 0]
  cy = s * x_t + c * y_t + cont.ref_origin_xy[:, 1]
  phi, _ = query_open_box_distance(
    pts,
    present=cont.present,
    cx=cx,
    cy=cy,
    z_b=cont.container[:, 2],
    inner_x=cont.container[:, 3],
    inner_y=cont.container[:, 4],
    h_c=cont.container[:, 5],
    t_wall=cont.container[:, 8],
    d_max=d_max,
  )
  r = torch.tensor(radii, device=env.device, dtype=phi.dtype).view(1, -1)
  clearance = (phi - r) / float(d_max)
  return clearance


def ee_target_error(
  env: ManagerBasedRlEnv,
  grasp_command_name: str = "grasp",
) -> torch.Tensor:
  """World-frame EE→target error for active arm (3-D), zeros if no arm."""
  robot = env.scene["robot"]
  grasp = env.command_manager.get_term(grasp_command_name)
  bnames = list(robot.body_names)
  left_i = bnames.index("left_wrist_yaw_link") if "left_wrist_yaw_link" in bnames else bnames.index("left_elbow_link")
  right_i = bnames.index("right_wrist_yaw_link") if "right_wrist_yaw_link" in bnames else bnames.index("right_elbow_link")
  arm = grasp.target_arm.squeeze(-1)
  ee = torch.where(
    (arm < 1.5).unsqueeze(-1),
    robot.data.body_link_pos_w[:, left_i],
    robot.data.body_link_pos_w[:, right_i],
  )
  err = grasp.target_pos_w - ee
  active = (arm > 0.5).float().unsqueeze(-1)
  return err * active
