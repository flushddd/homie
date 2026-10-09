"""Observation helpers for HOMIE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def height_command(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> torch.Tensor:
  """Return the height channel of twist+height command as ``(N, 1)``."""
  cmd = env.command_manager.get_command(command_name)
  return cmd[:, 3:4]


def twist_command_xy_yaw(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> torch.Tensor:
  """Return ``[vx, vy, yaw]`` only (exclude height)."""
  cmd = env.command_manager.get_command(command_name)
  return cmd[:, :3]
