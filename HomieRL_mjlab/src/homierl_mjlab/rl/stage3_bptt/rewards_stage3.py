"""Differentiable stage-3 rewards from executed / predicted geometry.

Real collect and imagination must share the same term definitions + present gate.
"""

from __future__ import annotations

from typing import Any

import torch


def target_reward(palm_pos: torch.Tensor, goal_pos: torch.Tensor, sigma: float) -> torch.Tensor:
  """Near-field soft reach: ``exp(-‖p−g‖ / σ)``."""
  d = torch.linalg.norm(palm_pos - goal_pos, dim=-1)
  return torch.exp(-d / float(sigma))


def approach_reward(
  palm_pos: torch.Tensor,
  goal_pos: torch.Tensor,
  sigma_far: float,
) -> torch.Tensor:
  """Far-field soft approach (larger σ). Does not force monotonic approach."""
  d = torch.linalg.norm(palm_pos - goal_pos, dim=-1)
  return torch.exp(-d / max(float(sigma_far), 1e-6))


def soft_success_reward(
  palm_pos: torch.Tensor,
  goal_pos: torch.Tensor,
  threshold: float,
) -> torch.Tensor:
  """Continuous near-goal peak (BPTT-friendly); eval can still use hard threshold."""
  d = torch.linalg.norm(palm_pos - goal_pos, dim=-1)
  thr = max(float(threshold), 1e-6)
  return torch.exp(-((d / thr) ** 2))


def safe_reward(
  clearance: torch.Tensor,
  d_safe: float,
  *,
  d_max: float = 0.40,
) -> torch.Tensor:
  """``clearance``: ``(B, J)`` as ``(phi-r)/d_max`` (same as ``arm_clearance_field``).

  Uses **worst** sphere (not mean) so one link clipping the rim is not washed out.
  Note: deep penetration can make this ``< -1`` if clearance is not clipped upstream.
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
  """Huber-ish on action delta (effective subgoal / offset), soft beyond 0.2 m."""
  d = action - last_action
  # Quadratic core, linear tails — avoids hard clamp gradient death.
  abs_d = d.abs()
  quad = torch.clamp(abs_d, max=0.2)
  lin = abs_d - quad
  return -torch.sum(quad * quad + 0.4 * lin, dim=-1)


def tilt_angle_from_root_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Angle (rad) between body +Z and world +Z from root quaternion ``(w,x,y,z)``."""
  w, x, y, z = quat_wxyz.unbind(-1)
  # Body z-axis expressed in world.
  zz = 1.0 - 2.0 * (x * x + y * y)
  return torch.acos(zz.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def soft_stability_reward(
  tilt_rad: torch.Tensor,
  *,
  limit_rad: float,
  soft_rad: float,
) -> torch.Tensor:
  """0 near upright; approaches -1 as tilt exceeds limit (no hard truncate)."""
  soft = max(float(soft_rad), 1e-3)
  over = torch.relu(tilt_rad - float(limit_rad)) / soft
  return -torch.tanh(over)


def stage3_reward_terms(
  *,
  palm_pos: torch.Tensor,
  goal_pos: torch.Tensor,
  clearance: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  sigma_p: float = 0.08,
  sigma_far: float = 0.35,
  d_safe: float = 0.05,
  d_max: float = 0.40,
  reach_threshold: float = 0.03,
  w_target: float = 1.0,
  w_approach: float = 0.3,
  w_safe: float = 1.5,
  w_smooth: float = 0.05,
  w_margin: float = 0.5,
  w_soft_success: float = 0.2,
  w_stability: float = 0.0,
  stability: torch.Tensor | None = None,
  present: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """Return raw + weighted terms and total ``r`` (present-gated safe/margin)."""
  raw_target = target_reward(palm_pos, goal_pos, sigma_p)
  raw_approach = approach_reward(palm_pos, goal_pos, sigma_far)
  raw_safe = safe_reward(clearance, d_safe, d_max=d_max)
  raw_margin = margin_reward(clearance, d_safe, d_max=d_max)
  raw_smooth = smooth_reward(action, last_action)
  raw_succ = soft_success_reward(palm_pos, goal_pos, reach_threshold)

  if present is not None:
    p = present.reshape(raw_safe.shape[0]).float()
  else:
    p = torch.ones_like(raw_safe)
  gated_safe = raw_safe * p
  gated_margin = raw_margin * p

  w_target_t = float(w_target) * raw_target
  w_approach_t = float(w_approach) * raw_approach
  w_safe_t = float(w_safe) * gated_safe
  w_margin_t = float(w_margin) * gated_margin
  w_smooth_t = float(w_smooth) * raw_smooth
  w_succ_t = float(w_soft_success) * raw_succ

  r = w_target_t + w_approach_t + w_safe_t + w_margin_t + w_smooth_t + w_succ_t
  raw_stab = torch.zeros_like(r)
  w_stab_t = torch.zeros_like(r)
  if stability is not None and float(w_stability) != 0.0:
    raw_stab = stability
    w_stab_t = float(w_stability) * raw_stab
    r = r + w_stab_t

  return {
    "r": r,
    "present": p,
    "raw_target": raw_target,
    "raw_approach": raw_approach,
    "raw_safe": raw_safe,
    "raw_margin": raw_margin,
    "raw_smooth": raw_smooth,
    "raw_soft_success": raw_succ,
    "raw_stability": raw_stab,
    "w_target": w_target_t,
    "w_approach": w_approach_t,
    "w_safe": w_safe_t,
    "w_margin": w_margin_t,
    "w_smooth": w_smooth_t,
    "w_soft_success": w_succ_t,
    "w_stability": w_stab_t,
  }


def stage3_reward(
  *,
  palm_pos: torch.Tensor,
  goal_pos: torch.Tensor,
  clearance: torch.Tensor,
  action: torch.Tensor,
  last_action: torch.Tensor,
  sigma_p: float = 0.08,
  sigma_far: float = 0.35,
  d_safe: float = 0.05,
  d_max: float = 0.40,
  reach_threshold: float = 0.03,
  w_target: float = 1.0,
  w_approach: float = 0.3,
  w_safe: float = 2.0,
  w_smooth: float = 0.05,
  w_margin: float = 0.5,
  w_soft_success: float = 0.2,
  w_stability: float = 0.0,
  stability: torch.Tensor | None = None,
  present: torch.Tensor | None = None,
  return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
  terms = stage3_reward_terms(
    palm_pos=palm_pos,
    goal_pos=goal_pos,
    clearance=clearance,
    action=action,
    last_action=last_action,
    sigma_p=sigma_p,
    sigma_far=sigma_far,
    d_safe=d_safe,
    d_max=d_max,
    reach_threshold=reach_threshold,
    w_target=w_target,
    w_approach=w_approach,
    w_safe=w_safe,
    w_smooth=w_smooth,
    w_margin=w_margin,
    w_soft_success=w_soft_success,
    w_stability=w_stability,
    stability=stability,
    present=present,
  )
  if return_terms:
    return terms["r"], terms
  return terms["r"]


def mean_terms(terms: dict[str, torch.Tensor], prefix: str = "") -> dict[str, float]:
  """Detach mean of tensor terms for logging."""
  out: dict[str, float] = {}
  for k, v in terms.items():
    if not isinstance(v, torch.Tensor) or v.numel() == 0:
      continue
    if v.ndim == 0:
      out[f"{prefix}{k}"] = float(v.detach().item())
    else:
      out[f"{prefix}{k}"] = float(v.detach().float().mean().cpu())
  return out


def present_split_means(
  terms: dict[str, torch.Tensor], key: str = "raw_safe"
) -> dict[str, float]:
  """Means of ``key`` conditioned on present / absent."""
  p = terms.get("present")
  v = terms.get(key)
  if p is None or v is None:
    return {}
  p = p.reshape(-1) > 0.5
  v = v.reshape(-1)
  out: dict[str, float] = {}
  if bool(p.any()):
    out[f"{key}_present"] = float(v[p].mean().detach().cpu())
  if bool((~p).any()):
    out[f"{key}_absent"] = float(v[~p].mean().detach().cpu())
  return out
