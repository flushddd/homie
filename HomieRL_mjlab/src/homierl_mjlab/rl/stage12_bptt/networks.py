"""Loco actor for Stage-1/2 SuperMimic (flat MLP; HIM history is future work)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

# Reuse Stage-3 WM / critic implementations (unchanged Stage-3 code).
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Critic as LocoCritic
from homierl_mjlab.rl.stage3_bptt.networks import WorldModel

__all__ = ["LocoActor", "LocoCritic", "WorldModel"]


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for h in hidden:
    layers += [nn.Linear(d, h), nn.SiLU()]
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


class LocoActor(nn.Module):
  """Deterministic tanh body action in [-1, 1]^D (Homie scale applied in env)."""

  def __init__(
    self,
    obs_dim: int,
    body_dim: int = 12,
    hidden: Sequence[int] = (256, 256, 128),
    *,
    near_zero_init: bool = True,
  ):
    super().__init__()
    self.body_dim = int(body_dim)
    self.net = _mlp(obs_dim, hidden, self.body_dim)
    if near_zero_init:
      last = self.net[-1]
      assert isinstance(last, nn.Linear)
      nn.init.zeros_(last.bias)
      with torch.no_grad():
        last.weight.mul_(0.01)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.net(obs))
