"""Stage-3 actor (body+EE), critic, and dynamics world model."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), nn.SiLU()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


class Stage3Actor(nn.Module):
  """Deterministic actor.

  Default ``body_ee_palm_delta`` with ``loco_obs_dim>0`` (v4):
    * ``body_net(loco_hist) → a_body(14)`` raw; ``loco_hist`` is HIM-aligned
      ``history_length × one_step`` (default 6×229), same entry as teacher
    * ``ee_net(reach_obs) → a_ee(3)`` meters Δp, hard-clipped;
      injected as ``p_sub`` (policy EE residual off)

  Obs layout must be ``[loco_hist | reach]``. Legacy ``loco_obs_dim=0`` keeps a
  single MLP on the full vector (reach-only or packed).
  """

  def __init__(
    self,
    obs_dim: int,
    hidden: Sequence[int] = (256, 256, 128),
    *,
    body_dim: int = 14,
    ee_dim: int = 3,
    ee_delta_scale_m: float = 0.03,
    action_scale_m: float = 0.02,
    control_mode: str = "body_ee_palm_delta",
    near_zero_body_init: bool = True,
    loco_obs_dim: int = 0,
    body_action_clip: float = 100.0,
  ):
    super().__init__()
    self.control_mode = str(control_mode)
    self.body_dim = int(body_dim)
    self.ee_dim = int(ee_dim)
    self.ee_delta_scale_m = float(ee_delta_scale_m)
    self.action_scale_m = float(action_scale_m)
    self.body_action_clip = float(body_action_clip)
    self.obs_dim = int(obs_dim)
    self.loco_obs_dim = max(0, int(loco_obs_dim))
    self.reach_obs_dim = max(0, self.obs_dim - self.loco_obs_dim)

    if self.control_mode == "body_ee_palm_delta":
      self.out_dim = self.body_dim + self.ee_dim
    else:
      self.out_dim = 3

    self.body_net: nn.Module | None = None
    self.ee_net: nn.Module | None = None
    self.net: nn.Module | None = None

    if (
      self.control_mode == "body_ee_palm_delta"
      and self.loco_obs_dim > 0
      and self.reach_obs_dim > 0
    ):
      self.body_net = _mlp(self.loco_obs_dim, hidden, self.body_dim)
      self.ee_net = _mlp(self.reach_obs_dim, hidden, self.ee_dim)
      if near_zero_body_init:
        last_b = self.body_net[-1]
        last_e = self.ee_net[-1]
        assert isinstance(last_b, nn.Linear) and isinstance(last_e, nn.Linear)
        nn.init.zeros_(last_b.bias)
        nn.init.zeros_(last_e.bias)
        with torch.no_grad():
          last_b.weight.mul_(0.01)
          last_e.weight.mul_(0.1)
    else:
      self.net = _mlp(obs_dim, hidden, self.out_dim)
      if self.control_mode == "body_ee_palm_delta" and near_zero_body_init:
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
          last.weight[: self.body_dim].mul_(0.01)
          last.weight[self.body_dim :].mul_(0.1)

  def _clip_body(self, a_body: torch.Tensor) -> torch.Tensor:
    c = float(self.body_action_clip)
    if c > 0.0:
      return a_body.clamp(-c, c)
    return a_body

  def _clip_ee(self, a_ee: torch.Tensor) -> torch.Tensor:
    """Meters in base; hard clip (oracle-style inject, no tanh×scale)."""
    s = float(self.ee_delta_scale_m)
    if s > 0.0:
      return a_ee.clamp(-s, s)
    return a_ee

  def forward(
    self, obs: torch.Tensor
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if self.body_net is not None and self.ee_net is not None:
      loco = obs[..., : self.loco_obs_dim]
      reach = obs[..., self.loco_obs_dim :]
      # Raw HIM joint actions (phase-1/2); no tanh.
      a_body = self._clip_body(self.body_net(loco))
      # Palm Δp in meters → p_sub inject; residual path unused.
      a_ee = self._clip_ee(self.ee_net(reach))
      return a_body, a_ee
    assert self.net is not None
    raw = self.net(obs)
    if self.control_mode != "body_ee_palm_delta":
      # Legacy hierarchical / residual: still clip to ±action_scale_m.
      return self._clip_ee_legacy(raw)
    a_body = self._clip_body(raw[..., : self.body_dim])
    a_ee = self._clip_ee(raw[..., self.body_dim :])
    return a_body, a_ee

  def _clip_ee_legacy(self, raw: torch.Tensor) -> torch.Tensor:
    s = float(self.action_scale_m)
    if s > 0.0:
      return raw.clamp(-s, s)
    return raw

  def forward_packed(self, obs: torch.Tensor) -> torch.Tensor:
    """Return ``(B, out_dim)`` packed action (body ‖ ee_delta meters)."""
    out = self.forward(obs)
    if isinstance(out, tuple):
      return torch.cat(out, dim=-1)
    return out


class Stage3Critic(nn.Module):
  """State-value critic conditioned on goal / geometry features."""

  def __init__(self, feat_dim: int, hidden: Sequence[int] = (256, 256, 128)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, 1)

  def forward(self, feat: torch.Tensor) -> torch.Tensor:
    return self.net(feat).squeeze(-1)


class WorldModel(nn.Module):
  """Deterministic next-state residual model: ``s' = s + clip(f(s, u, ctx))``.

  ``u`` is the full applied joint-position command (same semantics as PD
  targets in sim). ``ctx`` holds static goal/geometry encodings.
  Residual clamp slows long-horizon explosion that often ends in NaN.
  """

  def __init__(
    self,
    state_dim: int,
    control_dim: int,
    ctx_dim: int = 0,
    hidden: Sequence[int] = (512, 512, 256),
    residual_clip: float = 0.5,
  ):
    super().__init__()
    self.state_dim = int(state_dim)
    self.control_dim = int(control_dim)
    self.ctx_dim = int(ctx_dim)
    self.residual_clip = float(residual_clip)
    self.net = _mlp(state_dim + control_dim + ctx_dim, hidden, state_dim)

  def forward(
    self,
    state: torch.Tensor,
    control: torch.Tensor,
    ctx: torch.Tensor | None = None,
  ) -> torch.Tensor:
    state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    control = torch.nan_to_num(control, nan=0.0, posinf=0.0, neginf=0.0)
    if ctx is None:
      if self.ctx_dim > 0:
        ctx = torch.zeros(
          state.shape[0], self.ctx_dim, device=state.device, dtype=state.dtype
        )
      else:
        x = torch.cat((state, control), dim=-1)
        delta = self.net(x)
        return state + delta.clamp(-self.residual_clip, self.residual_clip)
    ctx = torch.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.cat((state, control, ctx), dim=-1)
    delta = self.net(x)
    return state + delta.clamp(-self.residual_clip, self.residual_clip)

  def rollout(
    self,
    state0: torch.Tensor,
    controls: torch.Tensor,
    ctx: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Open-loop autoregressive rollout.

    Args:
      state0: ``(B, S)``
      controls: ``(H, B, U)`` or ``(B, H, U)``
    Returns:
      states ``(H+1, B, S)`` including ``state0``.
    """
    if controls.dim() == 3 and controls.shape[1] == state0.shape[0]:
      seq = controls
    else:
      seq = controls.transpose(0, 1)
    h = seq.shape[0]
    states = [state0]
    s = state0
    for t in range(h):
      s = self.forward(s, seq[t], ctx)
      states.append(s)
    return torch.stack(states, dim=0)
