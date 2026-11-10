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
  present: torch.Tensor
  episode_id: torch.Tensor
  env_id: torch.Tensor


class EpisodeReplay:
  """Fixed-capacity circular buffer; window samples stay within one episode.

  Collector writes all ``N`` envs each step as one block. Capacity **must** be a
  multiple of ``N`` so wrap never splits a step (otherwise env stride breaks).

  Stores per-step ``present`` so imagination can match real safe/margin gating.
  Old checkpoints without ``present`` are not auto-migrated — rebuild replay.
  """

  def __init__(
    self,
    capacity: int,
    state_dim: int,
    control_dim: int,
    goal_dim: int,
    geom_dim: int,
    device: torch.device,
    *,
    num_envs: int = 1,
  ):
    self.num_envs = max(1, int(num_envs))
    # Keep stride layout valid across wrap.
    cap = max(int(capacity), self.num_envs * 16)
    if cap % self.num_envs != 0:
      cap = (cap // self.num_envs + 1) * self.num_envs
    self.capacity = cap
    self.device = device
    self.ptr = 0
    self.size = 0
    self._steps_written = 0
    z = lambda d: torch.zeros(cap, d, device=device)
    self.state = z(state_dim)
    self.control = z(control_dim)
    self.next_state = z(state_dim)
    self.reward = torch.zeros(cap, device=device)
    self.terminated = torch.zeros(cap, device=device)
    self.truncated = torch.zeros(cap, device=device)
    self.goal = z(goal_dim)
    self.geom = z(geom_dim)
    self.present = torch.zeros(cap, device=device)
    self.episode_id = torch.zeros(cap, device=device, dtype=torch.long)
    self.env_id = torch.zeros(cap, device=device, dtype=torch.long)
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
    present: torch.Tensor | None = None,
    new_episode: torch.Tensor | None = None,
  ) -> None:
    del new_episode  # kept for API; prefer explicit episode_id
    n = int(state.shape[0])
    if n != self.num_envs:
      raise ValueError(
        f"add_batch n={n} != replay.num_envs={self.num_envs}; "
        "keep num_envs fixed for stride window sampling"
      )
    if self.ptr % self.num_envs != 0:
      raise RuntimeError("replay ptr desynced from env block stride")
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
    if present is None:
      self.present[slots] = 0.0
    else:
      self.present[slots] = present.reshape(n).float()
    self.env_id[slots] = env_id.long()
    if episode_id is None:
      self.episode_id[slots] = self._ep_counter
      self._ep_counter += 1
    else:
      # Pack env-local episode with env_id so windows never cross envs.
      self.episode_id[slots] = env_id.long() * 1_000_000_000 + episode_id.long()
    self.ptr = int((self.ptr + n) % self.capacity)
    self.size = min(self.capacity, self.size + n)
    self._steps_written += 1

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
      present=self.present[idx],
      episode_id=self.episode_id[idx],
      env_id=self.env_id[idx],
    )

  def _num_complete_steps(self) -> int:
    return int(self.size // self.num_envs)

  def _chrono_index(self, step: torch.Tensor, env_i: int) -> torch.Tensor:
    """Map chronological step (0=oldest kept) + env → absolute slot.

    When not full, data starts at 0. When full, oldest step starts at ``ptr``.
    """
    n = self.num_envs
    if self.size < self.capacity:
      return step * n + env_i
    # Full circular: chronological step 0 begins at ptr.
    return (self.ptr + step * n + env_i) % self.capacity

  def sample_contiguous_windows(
    self, batch: int, horizon: int, num_envs_hint: int | None = None
  ) -> tuple[torch.Tensor, ...]:
    """Return states/controls/.../present from same env episode.

    Returns:
      ``states (B,H+1,S), controls (B,H,U), rewards (B,H), terminated (B,H),
      truncated (B,H), goals (B,G), geoms (B,Geom), present (B,)``
      ``present`` / ``goals`` / ``geoms`` are taken from the window start
      (static within an episode for container curriculum).
    """
    h = int(horizon)
    n_envs = int(num_envs_hint) if num_envs_hint is not None else self.num_envs
    if n_envs != self.num_envs:
      # Hint mismatch (e.g. stale arg): trust the buffer stride.
      n_envs = self.num_envs
    n_steps = self._num_complete_steps()
    if n_steps <= h + 1:
      raise RuntimeError(
        "Replay too short for contiguous windows "
        f"(steps={n_steps}, need>{h + 1}, size={self.size}, "
        f"n_envs={n_envs}, capacity={self.capacity}). "
        "Increase --warmup-collect / replay_capacity, or lower --num-envs."
      )

    states, controls, rewards, terms, truncs, goals, geoms, presents = (
      [],
      [],
      [],
      [],
      [],
      [],
      [],
      [],
    )
    tries = 0
    max_start = n_steps - h  # exclusive end for start step
    while len(states) < batch and tries < batch * 100:
      tries += 1
      env_i = int(torch.randint(0, n_envs, (1,)).item())
      t0 = int(torch.randint(0, max_start, (1,)).item())
      steps = torch.arange(h + 1, device=self.device) + t0
      idx = self._chrono_index(steps, env_i)
      ep = self.episode_id[idx]
      if not bool((ep == ep[0]).all()):
        continue
      if not bool((self.env_id[idx] == env_i).all()):
        continue
      # No terminal inside the open prefix (keep last step free for bootstrap).
      # Truncation (time_out) is allowed — critic still bootstraps.
      if h > 1 and bool(self.terminated[idx[:-2]].any()):
        continue
      states.append(self.state[idx])
      controls.append(self.control[idx[:-1]])
      rewards.append(self.reward[idx[:-1]])
      terms.append(self.terminated[idx[:-1]])
      truncs.append(self.truncated[idx[:-1]])
      goals.append(self.goal[idx[0]])
      geoms.append(self.geom[idx[0]])
      presents.append(self.present[idx[0]])
    if not states:
      raise RuntimeError(
        "Failed to sample contiguous replay windows "
        f"(size={self.size}, steps={n_steps}, n_envs={n_envs}, h={h}, "
        f"capacity={self.capacity}). Episodes may be shorter than h+1."
      )
    return (
      torch.stack(states, 0),
      torch.stack(controls, 0),
      torch.stack(rewards, 0),
      torch.stack(terms, 0),
      torch.stack(truncs, 0),
      torch.stack(goals, 0),
      torch.stack(geoms, 0),
      torch.stack(presents, 0),
    )
