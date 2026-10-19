"""Analytic container distance field (ground-truth, batched).

Convention (Stage3 doc): free space ``phi > 0``, surface ``0``, material ``phi < 0``.
Open-bottom square walls = union of four thin boxes; cavity is free space.
"""

from __future__ import annotations

import math

import torch


def sdf_axis_aligned_box(
  points: torch.Tensor, center: torch.Tensor, half_sizes: torch.Tensor
) -> torch.Tensor:
  """Euclidean SDF of axis-aligned boxes.

  Args:
    points: ``(N, K, 3)`` or ``(N, 3)``
    center: ``(N, 3)``
    half_sizes: ``(N, 3)``
  Returns:
    ``(N, K)`` or ``(N,)`` signed distances.
  """
  squeeze = points.ndim == 2
  if squeeze:
    points = points.unsqueeze(1)
  c = center.unsqueeze(1)
  h = half_sizes.unsqueeze(1)
  q = (points - c).abs() - h
  outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
  inside = torch.clamp(q.max(dim=-1).values, max=0.0)
  phi = outside + inside
  return phi.squeeze(1) if squeeze else phi


def open_box_wall_centers_half(
  *,
  cx: torch.Tensor,
  cy: torch.Tensor,
  z_b: torch.Tensor,
  inner_x: torch.Tensor,
  inner_y: torch.Tensor,
  h_c: torch.Tensor,
  t_wall: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Four upright wall boxes for an open-bottom square container.

  Returns:
    centers ``(N, 4, 3)``, half_sizes ``(N, 4, 3)`` in world/task frame
    (same XY as ``cx,cy``; Z up from ``z_b``).
  """
  n = cx.shape[0]
  device = cx.device
  dtype = cx.dtype
  # Outer extents.
  hx = 0.5 * inner_x + t_wall
  hy = 0.5 * inner_y + t_wall
  z_c = z_b + 0.5 * h_c
  half_h = 0.5 * h_c
  # +X, -X, +Y, -Y walls (thin in normal direction).
  centers = torch.zeros(n, 4, 3, device=device, dtype=dtype)
  halves = torch.zeros(n, 4, 3, device=device, dtype=dtype)
  # +X wall
  centers[:, 0, 0] = cx + 0.5 * inner_x + 0.5 * t_wall
  centers[:, 0, 1] = cy
  centers[:, 0, 2] = z_c
  halves[:, 0, 0] = 0.5 * t_wall
  halves[:, 0, 1] = hy
  halves[:, 0, 2] = half_h
  # -X wall
  centers[:, 1, 0] = cx - 0.5 * inner_x - 0.5 * t_wall
  centers[:, 1, 1] = cy
  centers[:, 1, 2] = z_c
  halves[:, 1, 0] = 0.5 * t_wall
  halves[:, 1, 1] = hy
  halves[:, 1, 2] = half_h
  # +Y wall
  centers[:, 2, 0] = cx
  centers[:, 2, 1] = cy + 0.5 * inner_y + 0.5 * t_wall
  centers[:, 2, 2] = z_c
  halves[:, 2, 0] = hx
  halves[:, 2, 1] = 0.5 * t_wall
  halves[:, 2, 2] = half_h
  # -Y wall
  centers[:, 3, 0] = cx
  centers[:, 3, 1] = cy - 0.5 * inner_y - 0.5 * t_wall
  centers[:, 3, 2] = z_c
  halves[:, 3, 0] = hx
  halves[:, 3, 1] = 0.5 * t_wall
  halves[:, 3, 2] = half_h
  return centers, halves


def query_open_box_distance(
  points_w: torch.Tensor,
  *,
  present: torch.Tensor,
  cx: torch.Tensor,
  cy: torch.Tensor,
  z_b: torch.Tensor,
  inner_x: torch.Tensor,
  inner_y: torch.Tensor,
  h_c: torch.Tensor,
  t_wall: torch.Tensor,
  d_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Batched distance to open-bottom square walls.

  Args:
    points_w: ``(N, K, 3)`` world points
    present: ``(N,)`` bool/float — if false, return ``+d_max`` and valid=1
  Returns:
    phi ``(N, K)``, valid ``(N, K)``
  """
  n, k, _ = points_w.shape
  device = points_w.device
  dtype = points_w.dtype
  centers, halves = open_box_wall_centers_half(
    cx=cx, cy=cy, z_b=z_b, inner_x=inner_x, inner_y=inner_y, h_c=h_c, t_wall=t_wall
  )
  # Min over 4 walls.
  phi = torch.full((n, k), float(d_max), device=device, dtype=dtype)
  for i in range(4):
    phi_i = sdf_axis_aligned_box(points_w, centers[:, i], halves[:, i])
    phi = torch.minimum(phi, phi_i)
  present_f = present.reshape(n, 1).to(dtype=dtype)
  phi = torch.where(present_f > 0.5, phi, torch.full_like(phi, float(d_max)))
  phi = torch.clamp(phi, -float(d_max), float(d_max))
  valid = torch.ones(n, k, device=device, dtype=dtype)
  return phi, valid


def make_shoulder_query_grid(
  *,
  count: int,
  x_range: tuple[float, float],
  y_range: tuple[float, float],
  z_range: tuple[float, float],
  device: torch.device,
  dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Deterministic ``(K, 3)`` query points in shoulder frame (forward/left/up)."""
  # Factor count into a near-cubic grid.
  nx = max(1, int(round(count ** (1.0 / 3.0))))
  ny = nx
  nz = max(1, int(math.ceil(count / float(nx * ny))))
  xs = torch.linspace(x_range[0], x_range[1], nx, device=device, dtype=dtype)
  ys = torch.linspace(y_range[0], y_range[1], ny, device=device, dtype=dtype)
  zs = torch.linspace(z_range[0], z_range[1], nz, device=device, dtype=dtype)
  grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
  if grid.shape[0] >= count:
    return grid[:count].contiguous()
  pad = grid[-1:].repeat(count - grid.shape[0], 1)
  return torch.cat([grid, pad], dim=0)
