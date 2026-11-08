"""Episode-aware replay for world-model windows and BPTT starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransitionBatch:
  state: torch.Tensor
  control: torch.Tensor
  next_state: torch.Tensor
  reward: torch.Tensor
  terminated: torch.Tensor
  truncated: torch.Tensor
  goal: torch.Tensor
  geom: torch.Tensor
  episode_id: torch.Tensor
  env_id: torch.Tensor


class EpisodeReplay:
  """Fixed-capacity circular buffer; window samples stay within one episode."""

  def __init__(
    self,
    capacity: int,
    state_dim: int,
    control_dim: int,
    goal_dim: int,
    geom_dim: int,
    device: torch.device,
  ):
    self.capacity = int(capacity)
    self.device = device
    self.ptr = 0
    self.size = 0
    z = lambda d: torch.zeros(capacity, d, device=device)
    self.state = z(state_dim)
    self.control = z(control_dim)
    self.next_state = z(state_dim)
    self.reward = torch.zeros(capacity, device=device)
    self.terminated = torch.zeros(capacity, device=device)
    self.truncated = torch.zeros(capacity, device=device)
    self.goal = z(goal_dim)
    self.geom = z(geom_dim)
    self.episode_id = torch.zeros(capacity, device=device, dtype=torch.long)
    self.env_id = torch.zeros(capacity, device=device, dtype=torch.long)
    self._ep_counter = 0

  def add_batch(
    self,
    *,
    state: torch.Tensor,
    control: torch.Tensor,
    next_state: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    goal: torch.Tensor,
    geom: torch.Tensor,
    env_id: torch.Tensor,
    episode_id: torch.Tensor | None = None,
    new_episode: torch.Tensor | None = None,
  ) -> None:
    del new_episode  # kept for API; prefer explicit episode_id
    n = state.shape[0]
    idx = torch.arange(n, device=self.device)
    slots = (self.ptr + idx) % self.capacity
    self.state[slots] = state
    self.control[slots] = control
    self.next_state[slots] = next_state
    self.reward[slots] = reward
    self.terminated[slots] = terminated.float()
    self.truncated[slots] = truncated.float()
    self.goal[slots] = goal
    self.geom[slots] = geom
    self.env_id[slots] = env_id.long()
    if episode_id is None:
      self.episode_id[slots] = self._ep_counter
      self._ep_counter += 1
    else:
      # Pack env-local episode with env_id so windows never cross envs.
      self.episode_id[slots] = env_id.long() * 1_000_000_000 + episode_id.long()
    self.ptr = int((self.ptr + n) % self.capacity)
    self.size = min(self.capacity, self.size + n)

  def sample_transitions(self, batch: int) -> TransitionBatch:
    assert self.size > 0
    idx = torch.randint(0, self.size, (batch,), device=self.device)
    return TransitionBatch(
      state=self.state[idx],
      control=self.control[idx],
      next_state=self.next_state[idx],
      reward=self.reward[idx],
      terminated=self.terminated[idx],
      truncated=self.truncated[idx],
      goal=self.goal[idx],
      geom=self.geom[idx],
      episode_id=self.episode_id[idx],
      env_id=self.env_id[idx],
    )

  def sample_contiguous_windows(
    self, batch: int, horizon: int, num_envs_hint: int | None = None
  ) -> tuple[torch.Tensor, ...]:
    """Return ``states (B,H+1,S), controls (B,H,U), ...`` from same env episode.

    Collector writes all envs each step as a contiguous block of size ``N``.
    Temporal neighbors for one env are therefore ``i, i+N, i+2N, ...``.
    """
    h = int(horizon)
    assert self.size > h + 1
    # Infer N from unique env ids in the filled region when possible.
    filled = min(self.size, self.capacity)
    if num_envs_hint is not None:
      n_envs = int(num_envs_hint)
    else:
      n_envs = int(self.env_id[:filled].max().item()) + 1
      n_envs = max(1, n_envs)
    states, controls, rewards, terms, goals, geoms = [], [], [], [], [], []
    tries = 0
    max_block = filled // n_envs
    while len(states) < batch and tries < batch * 80:
      tries += 1
      if max_block <= h + 1:
        break
      env_i = int(torch.randint(0, n_envs, (1,)).item())
      t0 = int(torch.randint(0, max_block - h, (1,)).item())
      idx = env_i + (torch.arange(h + 1, device=self.device) + t0) * n_envs
      if int(idx[-1].item()) >= filled:
        continue
      # Avoid circular wrap corruption.
      if int(idx[-1].item()) >= self.capacity:
        continue
      ep = self.episode_id[idx]
      if not bool((ep == ep[0]).all()):
        continue
      if not bool((self.env_id[idx] == env_i).all()):
        continue
      if h > 1 and bool(self.terminated[idx[:-2]].any()):
        continue
      states.append(self.state[idx])
      controls.append(self.control[idx[:-1]])
      rewards.append(self.reward[idx[:-1]])
      terms.append(self.terminated[idx[:-1]])
      goals.append(self.goal[idx[0]])
      geoms.append(self.geom[idx[0]])
    if not states:
      raise RuntimeError(
        "Failed to sample contiguous replay windows "
        f"(size={self.size}, n_envs={n_envs}, h={h})"
      )
    return (
      torch.stack(states, 0),
      torch.stack(controls, 0),
      torch.stack(rewards, 0),
      torch.stack(terms, 0),
      torch.stack(goals, 0),
      torch.stack(geoms, 0),
    )
