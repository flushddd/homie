"""Palm / hand TCP relative to Inspire ``wrist_yaw_link``.

The FTP MJCF has no separate palm body — the palm force-sensor geom sits at a
fixed offset on ``*_wrist_yaw_link``. Reach IK and EE error use that point so
the **hand** (not the wrist joint origin) tracks the grasp target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.entity import Entity

# From ``g1_29dof_rev_1_0_with_inspire_hand_FTP.xml`` palm_force_sensor geom.
LEFT_PALM_OFFSET_LOCAL = (0.1611, -0.012296, 0.00036819)
RIGHT_PALM_OFFSET_LOCAL = (0.1611, 0.012296, 0.00036819)
LEFT_EE_LINK = "left_wrist_yaw_link"
RIGHT_EE_LINK = "right_wrist_yaw_link"


def palm_offset_tensor(
  device: torch.device,
  dtype: torch.dtype,
  *,
  left: bool,
) -> torch.Tensor:
  off = LEFT_PALM_OFFSET_LOCAL if left else RIGHT_PALM_OFFSET_LOCAL
  return torch.tensor(off, device=device, dtype=dtype)


def link_point_pos_w(
  robot: Entity,
  body_id: int,
  offset_local: torch.Tensor,
  env_ids: torch.Tensor | slice | None = None,
) -> torch.Tensor:
  """World position of a fixed point on a body (``p + R * offset_local``)."""
  if env_ids is None:
    pos = robot.data.body_link_pos_w[:, body_id]
    quat = robot.data.body_link_quat_w[:, body_id]
  else:
    pos = robot.data.body_link_pos_w[env_ids, body_id]
    quat = robot.data.body_link_quat_w[env_ids, body_id]
  off = offset_local.to(device=pos.device, dtype=pos.dtype)
  if off.ndim == 1:
    off = off.unsqueeze(0).expand(pos.shape[0], 3)
  return pos + quat_apply(quat, off)
