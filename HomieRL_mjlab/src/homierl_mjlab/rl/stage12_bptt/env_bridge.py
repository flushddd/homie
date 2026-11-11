"""State packing for Stage-1/2 BPTT (no EE residual)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def pack_root_state(robot) -> torch.Tensor:
  return torch.cat(
    (
      robot.data.root_link_pos_w,
      robot.data.root_link_quat_w,
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
    ),
    dim=-1,
  )


def pack_robot_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """``q | qd | root(pos3+quat4+lin3+ang3)`` — loco WM state (no palm)."""
  robot = env.scene["robot"]
  return torch.cat(
    (robot.data.joint_pos, robot.data.joint_vel, pack_root_state(robot)),
    dim=-1,
  )


def flatten_policy_obs(obs_td) -> torch.Tensor:
  """Prefer ``policy`` / ``actor`` TensorDict group; else flatten all tensors."""
  if hasattr(obs_td, "get"):
    for key in ("policy", "actor", "obs"):
      if key in obs_td.keys():
        group = obs_td.get(key)
        if hasattr(group, "keys"):
          parts = []
          for k in sorted(group.keys()):
            v = group.get(k)
            if isinstance(v, torch.Tensor):
              parts.append(v.reshape(v.shape[0], -1))
          if parts:
            return torch.cat(parts, dim=-1)
        if isinstance(group, torch.Tensor):
          return group.reshape(group.shape[0], -1)
  if isinstance(obs_td, torch.Tensor):
    return obs_td.reshape(obs_td.shape[0], -1)
  # Last resort: concatenate every tensor leaf.
  parts = []
  if hasattr(obs_td, "items"):
    for _, v in obs_td.items():
      if isinstance(v, torch.Tensor):
        parts.append(v.reshape(v.shape[0], -1))
  if not parts:
    raise RuntimeError("Could not flatten policy observations")
  return torch.cat(parts, dim=-1)
