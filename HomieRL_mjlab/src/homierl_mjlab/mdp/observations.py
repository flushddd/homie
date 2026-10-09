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
  absolute: bool = True,
  base_height_target: float = 0.74,
) -> torch.Tensor:
  """Height command as ``(N, 1)``.

  Homie puts **absolute** meters in obs (``0.74 + offset``). Set ``absolute=False``
  to observe the raw offset channel instead.
  """
  cmd = env.command_manager.get_command(command_name)
  h = cmd[:, 3:4]
  if absolute:
    return h + base_height_target
  return h


def twist_command_xy_yaw(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  scale: tuple[float, float, float] = (2.0, 2.0, 0.5),
) -> torch.Tensor:
  """Return ``[vx, vy, yaw]`` with Homie ``obs_scales`` (lin_vel=2, ang_vel=0.5)."""
  cmd = env.command_manager.get_command(command_name)
  s = torch.tensor(scale, device=cmd.device, dtype=cmd.dtype)
  return cmd[:, :3] * s
