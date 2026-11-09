"""Hierarchical Stage-3: actor proposes subgoals for the frozen loco+IK chain.

``p_goal_W`` — true task goal (reward / success); never overwritten by the actor.
``p_ee0_W`` — palm world position at episode start (per-env anchor).
``a_t`` — offset in **world** meters relative to ``p_ee0_W`` (not a step Δ from current palm).
``p_sub_W = rate_limit(p_ee0_W + a_t)`` — short-horizon target injected into grasp so
legs/waist/IK all chase it; EE residual is held at 0 (no double control).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from homierl_mjlab.mdp.base_frame import base_to_world, world_to_base
from homierl_mjlab.mdp.palm import (
  LEFT_EE_LINK,
  RIGHT_EE_LINK,
  link_point_pos_w,
  palm_offset_tensor,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def palm_pos_world(env: ManagerBasedRlEnv, *, arm_left: bool) -> torch.Tensor:
  robot = env.scene["robot"]
  ee_name = LEFT_EE_LINK if arm_left else RIGHT_EE_LINK
  ids, _ = robot.find_bodies(ee_name, preserve_order=True)
  off = palm_offset_tensor(env.device, robot.data.joint_pos.dtype, left=arm_left)
  env_ids = torch.arange(env.num_envs, device=env.device)
  return link_point_pos_w(robot, int(ids[0]), off, env_ids)


def decode_offset_w(
  raw: torch.Tensor,
  *,
  scale_m: float,
) -> torch.Tensor:
  """``tanh``-style raw already scaled by the actor → clip to ``±scale_m``."""
  s = float(scale_m)
  return torch.clamp(raw, -s, s)


def rate_limit(
  p_raw: torch.Tensor,
  p_prev: torch.Tensor,
  *,
  max_step_m: float,
) -> torch.Tensor:
  """Limit per-step change of the world subgoal."""
  d = p_raw - p_prev
  n = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
  step = float(max_step_m)
  scale = (step / n).clamp(max=1.0)
  return p_prev + d * scale


class HierarchicalSubgoalState:
  """Per-env anchors + previous subgoal for hierarchical Stage-3."""

  def __init__(self, num_envs: int, device: torch.device):
    self.device = device
    self.p_ee0_w = torch.zeros(num_envs, 3, device=device)
    self.p_goal_w = torch.zeros(num_envs, 3, device=device)
    self.p_sub_w = torch.zeros(num_envs, 3, device=device)
    self.last_offset_w = torch.zeros(num_envs, 3, device=device)
    self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def reset_envs(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    *,
    arm_left: bool,
  ) -> None:
    if env_ids.numel() == 0:
      return
    palm_w = palm_pos_world(env, arm_left=arm_left)[env_ids]
    grasp = env.command_manager.get_term("grasp")
    goal_w = grasp.target_pos_w[env_ids].clone()
    self.p_ee0_w[env_ids] = palm_w
    self.p_goal_w[env_ids] = goal_w
    # Start at the anchor (zero offset); lower stack sees current palm.
    self.p_sub_w[env_ids] = palm_w
    self.last_offset_w[env_ids] = 0.0
    self.initialized[env_ids] = True

  def ensure_all(
    self,
    env: ManagerBasedRlEnv,
    *,
    arm_left: bool,
  ) -> None:
    missing = (~self.initialized).nonzero(as_tuple=False).flatten()
    if missing.numel():
      self.reset_envs(env, missing, arm_left=arm_left)

  def apply_offset(
    self,
    offset_w: torch.Tensor,
    *,
    scale_m: float,
    rate_m: float,
  ) -> torch.Tensor:
    """Decode + rate-limit; update ``p_sub_w`` / ``last_offset_w``. Returns ``p_sub_w``."""
    delta = decode_offset_w(offset_w, scale_m=scale_m)
    raw = self.p_ee0_w + delta
    self.p_sub_w = rate_limit(raw, self.p_sub_w, max_step_m=rate_m)
    self.last_offset_w = self.p_sub_w - self.p_ee0_w
    return self.p_sub_w


def inject_subgoal_into_grasp(
  env: ManagerBasedRlEnv,
  p_sub_w: torch.Tensor,
  env_ids: torch.Tensor | None = None,
) -> None:
  """Write short-horizon target into grasp world + body caches (all consumers)."""
  grasp = env.command_manager.get_term("grasp")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids.numel() == 0:
    return
  grasp.target_pos_w[env_ids] = p_sub_w[env_ids]
  # Keep body-frame command / extend_dist / obs in sync with world target.
  if hasattr(grasp, "_reproject_base_from_anchor"):
    grasp._reproject_base_from_anchor(env_ids)
  if hasattr(grasp, "_update_extend_dist"):
    grasp._update_extend_dist()


def attractor_offset_w(
  env: ManagerBasedRlEnv,
  hier: HierarchicalSubgoalState,
  *,
  arm_left: bool,
  scale_m: float,
  clip_m: float,
) -> torch.Tensor:
  """Stage-A collector: offset from ``p_ee0`` toward true ``p_goal`` (world)."""
  palm_w = palm_pos_world(env, arm_left=arm_left)
  # Prefer going to the goal; express as offset from episode anchor.
  desired = hier.p_goal_w - hier.p_ee0_w
  # Also nudge from current palm error so early steps move.
  toward = hier.p_goal_w - palm_w
  blend = 0.7 * desired + 0.3 * (palm_w + toward - hier.p_ee0_w)
  step = torch.clamp(blend, -float(clip_m), float(clip_m))
  n = torch.linalg.norm(step, dim=-1, keepdim=True).clamp_min(1e-6)
  step = step * (float(scale_m) / n).clamp(max=1.0)
  return step


def build_hierarchical_obs(
  env: ManagerBasedRlEnv,
  hier: HierarchicalSubgoalState,
  *,
  arm_left: bool,
  sdf: torch.Tensor,
  clearance: torch.Tensor,
) -> torch.Tensor:
  """Actor obs: palm, true-goal, goal−ee0, palm−ee0, last offset, SDF, clearance."""
  robot = env.scene["robot"]
  env_ids = torch.arange(env.num_envs, device=env.device)
  palm_w = palm_pos_world(env, arm_left=arm_left)
  palm_b = world_to_base(robot, palm_w, env_ids)
  goal_b = world_to_base(robot, hier.p_goal_w, env_ids)
  # Directional offsets in base (rotate only) so policy is yaw-invariant.
  from homierl_mjlab.mdp.base_frame import rotate_world_to_base

  goal_m_ee0_b = rotate_world_to_base(robot, hier.p_goal_w - hier.p_ee0_w, env_ids)
  palm_m_ee0_b = rotate_world_to_base(robot, palm_w - hier.p_ee0_w, env_ids)
  last_off_b = rotate_world_to_base(robot, hier.last_offset_w, env_ids)
  return torch.cat(
    (palm_b, goal_b, goal_m_ee0_b, palm_m_ee0_b, last_off_b, sdf, clearance),
    dim=-1,
  )


def hierarchical_obs_dim(sdf_query_count: int) -> int:
  # palm3 + goal3 + (goal-ee0)3 + (palm-ee0)3 + last_off3 + sdfK + clr6
  return 3 + 3 + 3 + 3 + 3 + int(sdf_query_count) + 6
