"""Stage-3 actor (Δp), critic, and dynamics world model."""

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
  """Deterministic 3-D action (meters), ``tanh``-scaled.

  - ``ee_residual`` mode: Δp in base from current palm.
  - ``hierarchical_subgoal`` mode: world-frame offset from episode ``p_ee0``.
  Do not apply an extra tanh outside.
  """

  def __init__(
    self,
    obs_dim: int,
    hidden: Sequence[int] = (256, 256, 128),
    action_scale_m: float = 0.02,
  ):
    super().__init__()
    self.net = _mlp(obs_dim, hidden, 3)
    self.action_scale_m = float(action_scale_m)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.net(obs)) * self.action_scale_m


class Stage3Critic(nn.Module):
  """State-value critic conditioned on goal / geometry features."""

  def __init__(self, feat_dim: int, hidden: Sequence[int] = (256, 256, 128)):
    super().__init__()
    self.net = _mlp(feat_dim, hidden, 1)

  def forward(self, feat: torch.Tensor) -> torch.Tensor:
    return self.net(feat).squeeze(-1)


class WorldModel(nn.Module):
  """Deterministic next-state residual model: ``s' = s + f(s, u, ctx)``.

  ``u`` is the full applied joint-position command (same semantics as PD
  targets in sim). ``ctx`` holds static goal/geometry encodings.
  """

  def __init__(
    self,
    state_dim: int,
    control_dim: int,
    ctx_dim: int = 0,
    hidden: Sequence[int] = (512, 512, 256),
  ):
    super().__init__()
    self.state_dim = int(state_dim)
    self.control_dim = int(control_dim)
    self.ctx_dim = int(ctx_dim)
    self.net = _mlp(state_dim + control_dim + ctx_dim, hidden, state_dim)

  def forward(
    self,
    state: torch.Tensor,
    control: torch.Tensor,
    ctx: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if ctx is None:
      if self.ctx_dim > 0:
        ctx = torch.zeros(
          state.shape[0], self.ctx_dim, device=state.device, dtype=state.dtype
        )
      else:
        x = torch.cat((state, control), dim=-1)
        return state + self.net(x)
    x = torch.cat((state, control, ctx), dim=-1)
    return state + self.net(x)

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
      # (H, B, U)
      seq = controls
    else:
      # (B, H, U)
      seq = controls.transpose(0, 1)
    h = seq.shape[0]
    states = [state0]
    s = state0
    for t in range(h):
      s = self.forward(s, seq[t], ctx)
      states.append(s)
    return torch.stack(states, dim=0)
