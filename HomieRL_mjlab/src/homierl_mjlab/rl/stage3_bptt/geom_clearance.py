"""Analytic open-box clearance from packed container geom (imagination / BPTT).

``geom`` is the 10-D container command in the task/base frame
(``x,y,z_b,inner_x,inner_y,h,b_rect,b_circle,t_wall,t_bottom``).
When absent, sizes are ~0 and we treat ``present=0``.
"""

from __future__ import annotations

import torch

from homierl_mjlab.mdp.sdf_geometry import (
  make_shoulder_query_grid,
  query_open_box_distance,
)


def _present_from_geom(geom: torch.Tensor) -> torch.Tensor:
  # b_rect flag or non-zero inner size.
  return ((geom[..., 6] > 0.5) | (geom[..., 3].abs() > 1e-4)).float()


def open_box_phi(
  points_b: torch.Tensor,
  geom: torch.Tensor,
  *,
  d_max: float = 0.40,
) -> torch.Tensor:
  """``points_b`` ``(B,K,3)`` in same frame as geom XY → ``phi (B,K)``."""
  present = _present_from_geom(geom)
  return query_open_box_distance(
    points_b,
    present=present,
    cx=geom[..., 0],
    cy=geom[..., 1],
    z_b=geom[..., 2],
    inner_x=geom[..., 3].clamp_min(1e-3),
    inner_y=geom[..., 4].clamp_min(1e-3),
    h_c=geom[..., 5].clamp_min(1e-3),
    t_wall=geom[..., 8].clamp_min(1e-3),
    d_max=d_max,
  )[0]


def palm_probe_clearance(
  palm_b: torch.Tensor,
  geom: torch.Tensor,
  *,
  d_max: float = 0.40,
  palm_radius: float = 0.05,
) -> torch.Tensor:
  """Clearance at palm + two probes (up / back) so BPTT feels rim contact.

  Returns ``(B, 3)`` as ``(phi - r) / d_max`` (same scale as ``arm_clearance_field``).
  """
  b = palm_b.shape[0]
  device = palm_b.device
  dtype = palm_b.dtype
  # Probes approximate forearm approach without full FK in imagination.
  up = torch.tensor([0.0, 0.0, 0.08], device=device, dtype=dtype)
  back = torch.tensor([-0.08, 0.0, 0.04], device=device, dtype=dtype)
  pts = torch.stack(
    (palm_b, palm_b + up, palm_b + back),
    dim=1,
  )  # (B,3,3)
  phi = open_box_phi(pts, geom, d_max=d_max)
  r = torch.tensor(
    [palm_radius, 0.055, 0.055], device=device, dtype=dtype
  ).view(1, -1)
  return (phi - r) / float(d_max)


def sdf_query_from_geom(
  geom: torch.Tensor,
  *,
  query_count: int = 64,
  d_max: float = 0.40,
) -> torch.Tensor:
  """Normalized SDF field ``(B, K)`` from fixed base-frame query grid."""
  grid = make_shoulder_query_grid(
    count=query_count,
    x_range=(0.10, 0.70),
    y_range=(-0.45, 0.45),
    z_range=(-0.25, 0.35),
    device=geom.device,
    dtype=geom.dtype,
  )
  b = geom.shape[0]
  pts = grid.view(1, -1, 3).expand(b, -1, -1)
  phi = open_box_phi(pts, geom, d_max=d_max)
  return phi / float(d_max)
