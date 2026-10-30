"""MuJoCo open-top crate (4 walls + floor) for phase-3 collide + viz.

Looks like a closed rectangular bin (open top only), matching the Homie
desktop-crate reference: continuous walls meeting at corners + solid floor.

Fixed-base → mjlab auto-wraps as mocap. Pose/size are updated per env from
``ContainerGeometryCommand.sync_walls_to_sim``.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

WALL_GEOM_NAMES = (
  "wall_xp",
  "wall_xn",
  "wall_yp",
  "wall_yn",
)
BOTTOM_GEOM_NAME = "bottom"

# Default crate (inner cavity / wall / height) — must match sync / play defaults.
# Sized like a desktop open-top storage bin (cyan reference crate).
_INNER = 0.36
_T_WALL = 0.028
_H = 0.22
_T_BOTTOM = 0.016


def _closed_crate_layout(
  inner_x: float = _INNER,
  inner_y: float = _INNER,
  h_c: float = _H,
  t_wall: float = _T_WALL,
  t_bottom: float = _T_BOTTOM,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], tuple[float, float, float], tuple[float, float, float]]:
  """Wall local pos/half + bottom pos/half for a closed open-top crate.

  Y-walls span the full outer X width; X-walls span the full outer Y width so
  corners overlap and there are no gaps (unlike thin non-overlapping panels).
  """
  hx_out = 0.5 * inner_x + t_wall
  hy_out = 0.5 * inner_y + t_wall
  half_h = 0.5 * h_c
  half_t = 0.5 * t_wall
  z_mid = half_h
  # Order: +X, -X, +Y, -Y (same as WALL_GEOM_NAMES / sync_walls_to_sim).
  positions = [
    (0.5 * inner_x + half_t, 0.0, z_mid),
    (-(0.5 * inner_x + half_t), 0.0, z_mid),
    (0.0, 0.5 * inner_y + half_t, z_mid),
    (0.0, -(0.5 * inner_y + half_t), z_mid),
  ]
  halves = [
    (half_t, hy_out, half_h),
    (half_t, hy_out, half_h),
    (hx_out, half_t, half_h),
    (hx_out, half_t, half_h),
  ]
  bottom_pos = (0.0, 0.0, 0.5 * t_bottom)
  bottom_half = (hx_out, hy_out, 0.5 * t_bottom)
  return positions, halves, bottom_pos, bottom_half


CONTAINER_ENTITY_NAME = "container_walls"


def get_container_walls_spec() -> mujoco.MjSpec:
  """Square open-top crate: four walls + floor under one mocap root."""
  spec = mujoco.MjSpec()
  root = spec.worldbody.add_body(name="container")
  # Cyan/teal like the reference crate (opaque).
  rgba = (0.25, 0.78, 0.78, 1.0)
  positions, halves, bottom_pos, bottom_half = _closed_crate_layout()
  for name, pos, half in zip(WALL_GEOM_NAMES, positions, halves, strict=True):
    root.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=half,
      rgba=rgba,
    )
  root.add_geom(
    name=BOTTOM_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=bottom_pos,
    size=bottom_half,
    rgba=rgba,
  )
  return spec


_WALL_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(wall_.*|bottom)$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=0,
  friction=(0.6,),
  disable_other_geoms=True,
)


def get_container_walls_cfg() -> EntityCfg:
  """Phase-3 container walls entity (mocap; start buried until first sync)."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    spec_fn=get_container_walls_spec,
    collisions=(_WALL_COLLISION,),
  )


__all__ = [
  "BOTTOM_GEOM_NAME",
  "CONTAINER_ENTITY_NAME",
  "WALL_GEOM_NAMES",
  "get_container_walls_cfg",
  "get_container_walls_spec",
]
