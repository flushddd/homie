"""Homie base / heading frame helpers (pelvis origin + yaw-only).

Used by grasp reproject, EE residual IK, and SDF queries so sim matches
deployable proprio frames (no world localization required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.entity import Entity


def heading_quat(robot: Entity, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
  """Yaw-only quaternion (wxyz) from root link."""
  if env_ids is None:
    return yaw_quat(robot.data.root_link_quat_w)
  return yaw_quat(robot.data.root_link_quat_w[env_ids])


def pelvis_pos(robot: Entity, env_ids: torch.Tensor | slice | None = None) -> torch.Tensor:
  if env_ids is None:
    return robot.data.root_link_pos_w[:, :3]
  return robot.data.root_link_pos_w[env_ids, :3]


def world_to_base(
  robot: Entity,
  points_w: torch.Tensor,
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """``p_base = R_yaw^T (p_w - pelvis)``."""
  origin = pelvis_pos(robot, env_ids)
  heading = heading_quat(robot, env_ids)
  return quat_apply_inverse(heading, points_w - origin)


def base_to_world(
  robot: Entity,
  points_b: torch.Tensor,
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """``p_w = pelvis + R_yaw p_base``."""
  origin = pelvis_pos(robot, env_ids)
  heading = heading_quat(robot, env_ids)
  return origin + quat_apply(heading, points_b)


def rotate_world_to_base(
  robot: Entity,
  vecs_w: torch.Tensor,
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """Rotate vectors (no translation): ``R_yaw^T v``."""
  heading = heading_quat(robot, env_ids)
  return quat_apply_inverse(heading, vecs_w)
