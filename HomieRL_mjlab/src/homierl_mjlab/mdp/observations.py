"""Observation helpers for HOMIE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Unified actor one-step (all phases):
#   proprio68 | a17 | grasp7 | sdf128 | clearance6 | ee_err3 = 229
HOMIE_PROPRIO_DIM = 68
HOMIE_ACTION_OBS_DIM = 17
HOMIE_GRASP_DIM = 7
HOMIE_SDF_DIM = 128  # 64 phi + 64 valid
HOMIE_CLEARANCE_DIM = 6
HOMIE_EE_ERR_DIM = 3
HOMIE_ONE_STEP_OBS = (
  HOMIE_PROPRIO_DIM
  + HOMIE_ACTION_OBS_DIM
  + HOMIE_GRASP_DIM
  + HOMIE_SDF_DIM
  + HOMIE_CLEARANCE_DIM
  + HOMIE_EE_ERR_DIM
)
# grasp[1] = is_waist → 68 + 17 + 1 = 86
HOMIE_IS_WAIST_IDX = HOMIE_PROPRIO_DIM + HOMIE_ACTION_OBS_DIM + 1


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


def grasp_target_command(
  env: ManagerBasedRlEnv,
  command_name: str = "grasp",
) -> torch.Tensor:
  """Homie waist-grasp obs: ``[is_target, is_waist, extend_dist, tx, ty, tz, arm]``."""
  return env.command_manager.get_command(command_name)


def padded_last_action(
  env: ManagerBasedRlEnv,
  target_dim: int = HOMIE_ACTION_OBS_DIM,
) -> torch.Tensor:
  """``last_action`` zero-padded to ``target_dim`` (phase-1/2 → 17 slots)."""
  act = env.action_manager.action
  if act.shape[-1] >= target_dim:
    return act[:, :target_dim]
  pad = torch.zeros(
    act.shape[0], target_dim - act.shape[-1], device=act.device, dtype=act.dtype
  )
  return torch.cat((act, pad), dim=-1)


def zeros_vector(env: ManagerBasedRlEnv, dim: int) -> torch.Tensor:
  """Fixed zero observation block (phase-1/2 placeholders for phase-3 channels)."""
  return torch.zeros(env.num_envs, int(dim), device=env.device)


def container_geometry_command(
  env: ManagerBasedRlEnv,
  command_name: str = "container",
  scale: tuple[float, ...] | None = None,
) -> torch.Tensor:
  """10-D container geometry at obs tail.

  ``[x_c, y_c, z_b, s_x, s_y, h_c, b_rect, b_circle, t_wall, t_bottom]``.
  Shape flags stay in {0,1}; optional per-dim ``scale`` multiplies the vector
  (train/deploy must match). Default: identity (meters / raw flags).
  """
  cmd = env.command_manager.get_command(command_name)
  if scale is None:
    return cmd
  s = torch.tensor(scale, device=cmd.device, dtype=cmd.dtype)
  if s.numel() != cmd.shape[-1]:
    raise ValueError(f"container scale len {s.numel()} != cmd dim {cmd.shape[-1]}")
  return cmd * s
