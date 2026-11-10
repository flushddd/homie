"""Differentiable stage-3 rewards from executed / predicted geometry."""

from __future__ import annotations

import torch


def target_reward(palm_pos: torch.Tensor, goal_pos: torch.Tensor, sigma: float) -> torch.Tensor:
  d = torch.linalg.norm(palm_pos - goal_pos, dim=-1)
  return torch.exp(-d / float(sigma))


def safe_reward(
  clearance: torch.Tensor,
  d_safe: float,
  *,
  d_max: float = 0.40,
) -> torch.Tensor:
  """``clearance``: ``(B, J)`` as ``(phi-r)/d_max`` (same as ``arm_clearance_field``).

  Uses **worst** sphere (not mean) so one link clipping the rim is not washed out.
  """
  c_m = clearance * float(d_max)
  d_s = max(float(d_safe), 1e-6)
  viol_n = torch.clamp(d_s - c_m, min=0.0) / d_s
  return -torch.amax(viol_n * viol_n, dim=-1)


def margin_reward(
  clearance: torch.Tensor,
  d_safe: float,
  *,
  d_max: float = 0.40,
) -> torch.Tensor:
  """Dense shaping: encourage worst-sphere clearance above ``d_safe`` (not only penalize)."""
  c_m = clearance * float(d_max)
  worst = torch.amin(c_m, dim=-1)
  d_s = max(float(d_safe), 1e-6)
  # Saturate in [0, 1]: 0 at/under d_safe, 1 at ≥ 2*d_safe.
  return torch.clamp((worst - d_s) / d_s, 0.0, 1.0)


def smooth_reward(action: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
  # Clamp so hierarchical large offsets cannot dominate / explode the reward.
  d = torch.clamp(action - last_action, -0.2, 0.2)
  return -torch.sum(d * d, dim=-1)


def stage3_reward(
  *,
  palm_pos: torch.Tensor,
  goal_pos: torch.Tensor,
  clearance: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  sigma_p: float = 0.08,
  d_safe: float = 0.05,
  d_max: float = 0.40,
  w_target: float = 1.0,
  w_safe: float = 2.0,
  w_smooth: float = 0.05,
  w_margin: float = 0.5,
  w_stability: float = 0.0,
  stability: torch.Tensor | None = None,
  present: torch.Tensor | None = None,
) -> torch.Tensor:
  safe = safe_reward(clearance, d_safe, d_max=d_max)
  margin = margin_reward(clearance, d_safe, d_max=d_max)
  if present is not None:
    p = present.reshape(safe.shape[0]).float()
    safe = safe * p
    margin = margin * p
  r = (
    float(w_target) * target_reward(palm_pos, goal_pos, sigma_p)
    + float(w_safe) * safe
    + float(w_margin) * margin
    + float(w_smooth) * smooth_reward(action, last_action)
  )
  if stability is not None and float(w_stability) != 0.0:
    r = r + float(w_stability) * stability
  return r
