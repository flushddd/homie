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
  """``q | qd | root_pos3 | quat_wxyz4 | lin_b3 | ang_b3 | palm_b3``."""
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


def root_quat_wxyz_from_state(state: torch.Tensor) -> torch.Tensor:
  """Packed-state root quaternion ``(w,x,y,z)``.

  Layout ends with ``palm_b(3)``; quat sits at ``[..., -13:-9]`` (not ``-16:-12``,
  which wrongly mixes root_pos with quat.w).
  """
  return state[..., -13:-9]


def projected_gravity_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Body-frame gravity direction from root quat (matches mjlab projected_gravity)."""
  w, x, y, z = quat_wxyz.unbind(-1)
  # Rotate world -Z into body: R^T @ [0,0,-1].
  gx = 2.0 * (x * z - w * y)
  gy = 2.0 * (y * z + w * x)
  gz = 1.0 - 2.0 * (x * x + y * y)
  # Above is +Z body in world z-row; gravity is -world_Z in body:
  return torch.stack((-gx, -gy, -gz), dim=-1)


def splice_loco_proprio_from_state(
  loco: torch.Tensor,
  state: torch.Tensor,
  *,
  n_joints: int,
  default_q: torch.Tensor,
  ang_vel_scale: float = 0.5,
  joint_vel_scale: float = 0.05,
  n_obs_joints: int = 29,
) -> torch.Tensor:
  """Refresh HIM one-step proprio slots from predicted packed state.

  Keeps commands / last_action / grasp / sdf / clearance from ``loco`` (not
  reconstructed). Phase-2/3 actor prefix::

    twist3 | height1 | ang_vel3 | gravity3 | joint_pos29 | joint_vel29 | ...

  This is **partial** closed-loop imag support — not a full HIM obs rebuild.
  """
  if loco.shape[-1] < 68:
    return loco
  q = state[..., :n_joints]
  qd = state[..., n_joints : 2 * n_joints]
  ang = state[..., -6:-3]
  quat = root_quat_wxyz_from_state(state)
  grav = projected_gravity_from_quat_wxyz(quat)
  def_q = default_q.to(device=q.device, dtype=q.dtype)
  if def_q.dim() == 1:
    def_q = def_q.unsqueeze(0)
  n_jp = min(int(n_obs_joints), int(n_joints), 29)
  jp = q[..., :n_jp] - def_q[..., :n_jp]
  jv = qd[..., :n_jp] * float(joint_vel_scale)
  if n_jp < 29:
    pad_p = torch.zeros(*jp.shape[:-1], 29 - n_jp, device=jp.device, dtype=jp.dtype)
    pad_v = torch.zeros_like(pad_p)
    jp = torch.cat((jp, pad_p), dim=-1)
    jv = torch.cat((jv, pad_v), dim=-1)
  out = loco.clone()
  out[..., 4:7] = ang * float(ang_vel_scale)
  out[..., 7:10] = grav
  out[..., 10:39] = jp
  out[..., 39:68] = jv
  return out


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
