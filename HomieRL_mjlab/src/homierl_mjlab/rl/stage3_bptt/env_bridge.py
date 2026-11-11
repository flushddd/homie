"""Bridge helpers between mjlab Homie env and Stage3-BPTT tensors.

Stage A attractor: small Δp toward grasp target for data collection before
the actor is trained. Frozen stage-2 loco still comes from the env's joint_pos
action / teacher path when using the Phase-3 task; this bridge only fills the
3-D EE residual channel (or applies DLS offline).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from homierl_mjlab.mdp.base_frame import world_to_base
from homierl_mjlab.mdp.palm import (
  LEFT_EE_LINK,
  RIGHT_EE_LINK,
  link_point_pos_w,
  palm_offset_tensor,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def pack_root_state(robot) -> torch.Tensor:
  """``pos3 | quat_wxyz4 | lin_vel_b3 | ang_vel_b3``."""
  return torch.cat(
    (
      robot.data.root_link_pos_w,
      robot.data.root_link_quat_w,
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
    ),
    dim=-1,
  )


def pack_robot_state(env: ManagerBasedRlEnv, *, arm_left: bool) -> torch.Tensor:
  robot = env.scene["robot"]
  q = robot.data.joint_pos
  qd = robot.data.joint_vel
  root = pack_root_state(robot)
  ee_name = LEFT_EE_LINK if arm_left else RIGHT_EE_LINK
  ids, _ = robot.find_bodies(ee_name, preserve_order=True)
  ee_body = int(ids[0])
  off = palm_offset_tensor(env.device, q.dtype, left=arm_left)
  env_ids = torch.arange(env.num_envs, device=env.device)
  palm_w = link_point_pos_w(robot, ee_body, off, env_ids)
  palm_b = world_to_base(robot, palm_w, env_ids)
  return torch.cat((q, qd, root, palm_b), dim=-1)


def palm_pos_base(env: ManagerBasedRlEnv, *, arm_left: bool) -> torch.Tensor:
  robot = env.scene["robot"]
  ee_name = LEFT_EE_LINK if arm_left else RIGHT_EE_LINK
  ids, _ = robot.find_bodies(ee_name, preserve_order=True)
  off = palm_offset_tensor(env.device, robot.data.joint_pos.dtype, left=arm_left)
  env_ids = torch.arange(env.num_envs, device=env.device)
  palm_w = link_point_pos_w(robot, int(ids[0]), off, env_ids)
  return world_to_base(robot, palm_w, env_ids)


def attractor_delta(
  env: ManagerBasedRlEnv,
  *,
  arm_left: bool = False,
  scale: float = 0.015,
  clip: float = 0.02,
) -> torch.Tensor:
  """Stage-A safe collector: Δp toward grasp target in base frame."""
  grasp = env.command_manager.get_term("grasp")
  palm_b = palm_pos_base(env, arm_left=arm_left)
  # Grasp target world → base
  env_ids = torch.arange(env.num_envs, device=env.device)
  tgt_b = world_to_base(env.scene["robot"], grasp.target_pos_w, env_ids)
  err = tgt_b - palm_b
  step = torch.clamp(err, -float(clip), float(clip))
  # Soft scale so early data stays small
  n = torch.linalg.norm(step, dim=-1, keepdim=True).clamp_min(1e-6)
  step = step * (float(scale) / n).clamp(max=1.0)
  return step


def goal_geom_from_env(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
  """Goal = grasp target in base (3). Geom = container raw 10-D (may be zeros)."""
  grasp = env.command_manager.get_term("grasp")
  env_ids = torch.arange(env.num_envs, device=env.device)
  goal = world_to_base(env.scene["robot"], grasp.target_pos_w, env_ids)
  try:
    cont = env.command_manager.get_command("container")
  except Exception:  # noqa: BLE001
    cont = torch.zeros(env.num_envs, 10, device=env.device)
  return goal, cont
