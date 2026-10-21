"""MuJoCo open-box container (4 walls) for phase-3 collide + viz.

Fixed-base → mjlab auto-wraps as mocap. Pose/size are updated per env from
``ContainerGeometryCommand`` (see ``sync_container_walls_to_sim``).
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Geom order must match ``open_box_wall_centers_half`` (+X, -X, +Y, -Y).
WALL_GEOM_NAMES = (
  "wall_xp",
  "wall_xn",
  "wall_yp",
  "wall_yn",
)

# Placeholder half-extents; overwritten every resample.
_PLACEHOLDER_HALF = (0.01, 0.15, 0.08)
_PLACEHOLDER_POS = (
  (0.16, 0.0, 0.08),
  (-0.16, 0.0, 0.08),
  (0.0, 0.16, 0.08),
  (0.0, -0.16, 0.08),
)

# Scene entity key (distinct from the "container" command term).
CONTAINER_ENTITY_NAME = "container_walls"


def get_container_walls_spec() -> mujoco.MjSpec:
  """Open-bottom square: four thin boxes under one mocap root."""
  spec = mujoco.MjSpec()
  root = spec.worldbody.add_body(name="container")
  rgba = (0.55, 0.58, 0.65, 0.55)
  for name, pos in zip(WALL_GEOM_NAMES, _PLACEHOLDER_POS, strict=True):
    root.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=_PLACEHOLDER_HALF,
      rgba=rgba,
    )
  return spec


# Feet: contype=0, conaffinity=1 → walls need contype bit0 to collide.
_WALL_COLLISION = CollisionCfg(
  geom_names_expr=(r"^wall_.*$",),
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
  "CONTAINER_ENTITY_NAME",
  "WALL_GEOM_NAMES",
  "get_container_walls_cfg",
  "get_container_walls_spec",
]
