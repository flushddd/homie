"""History-stacking adapter from mjlab RslRlVecEnvWrapper → HIM flat tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict


class HomieHistoryAdapter:
  """Maintain Homie-style step-major observation history on top of mjlab obs."""

  def __init__(
    self,
    env,
    actor_history_length: int = 6,
    critic_history_length: int = 1,
  ):
    self.env = env
    self.actor_history_length = actor_history_length
    self.critic_history_length = critic_history_length
    self.device = env.device
    self.num_envs = env.num_envs
    self.num_actions = env.num_actions

    # Peek one-step dims.
    td = env.get_observations()
    actor_step = self._as_tensor(td["actor"])
    critic_step = self._as_tensor(td["critic"])
    self.num_one_step_obs = int(actor_step.shape[-1])
    self.num_one_step_privileged_obs = int(critic_step.shape[-1])
    self.num_obs = self.num_one_step_obs * actor_history_length
    self.num_privileged_obs = self.num_one_step_privileged_obs * critic_history_length
    self.num_lower_dof = self.num_actions

    self._actor_hist = torch.zeros(
      self.num_envs, self.num_obs, device=self.device
    )
    self._critic_hist = torch.zeros(
      self.num_envs, self.num_privileged_obs, device=self.device
    )
    self._push(actor_step, critic_step, reset_all=True)

  @staticmethod
  def _as_tensor(x: torch.Tensor | TensorDict) -> torch.Tensor:
    if isinstance(x, TensorDict):
      return torch.cat([x[k] for k in x.keys()], dim=-1)
    return x

  def _push(
    self,
    actor_step: torch.Tensor,
    critic_step: torch.Tensor,
    env_ids: torch.Tensor | None = None,
    reset_all: bool = False,
  ) -> None:
    if reset_all or env_ids is None:
      # Shift and append for all envs.
      if self.actor_history_length > 1:
        self._actor_hist[:, : -self.num_one_step_obs] = self._actor_hist[
          :, self.num_one_step_obs :
        ].clone()
      self._actor_hist[:, -self.num_one_step_obs :] = actor_step

      if self.critic_history_length > 1:
        self._critic_hist[:, : -self.num_one_step_privileged_obs] = self._critic_hist[
          :, self.num_one_step_privileged_obs :
        ].clone()
      self._critic_hist[:, -self.num_one_step_privileged_obs :] = critic_step
    else:
      ids = env_ids
      if self.actor_history_length > 1:
        self._actor_hist[ids, : -self.num_one_step_obs] = self._actor_hist[
          ids, self.num_one_step_obs :
        ].clone()
      self._actor_hist[ids, -self.num_one_step_obs :] = actor_step[ids]

      if self.critic_history_length > 1:
        self._critic_hist[ids, : -self.num_one_step_privileged_obs] = self._critic_hist[
          ids, self.num_one_step_privileged_obs :
        ].clone()
      self._critic_hist[ids, -self.num_one_step_privileged_obs :] = critic_step[ids]

  def get_observations(self) -> torch.Tensor:
    return self._actor_hist.clone()

  def get_privileged_observations(self) -> torch.Tensor:
    return self._critic_hist.clone()

  def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
    td, _extras = self.env.reset()
    actor_step = self._as_tensor(td["actor"])
    critic_step = self._as_tensor(td["critic"])
    # Fill history with the reset frame.
    for _ in range(self.actor_history_length):
      self._push(actor_step, critic_step, reset_all=True)
    return self.get_observations(), self.get_privileged_observations()

  def step(self, actions: torch.Tensor):
    obs_td, rew, dones, extras = self.env.step(actions)
    actor_step = self._as_tensor(obs_td["actor"])
    critic_step = self._as_tensor(obs_td["critic"])

    # Capture termination critic obs before history overwrite for terminated envs.
    term_ids = (dones > 0).nonzero(as_tuple=False).flatten()
    term_priv = critic_step[term_ids].clone() if len(term_ids) else critic_step[:0]

    self._push(actor_step, critic_step, reset_all=True)

    # For terminated envs, fill history with the post-reset frame.
    if len(term_ids):
      # After mjlab reset, obs_td already contains reset obs for those envs.
      for _ in range(self.actor_history_length - 1):
        self._push(actor_step, critic_step, env_ids=term_ids)

    # Build termination privileged obs with history length (Homie API).
    if len(term_ids):
      term_hist = torch.zeros(
        len(term_ids),
        self.num_privileged_obs,
        device=self.device,
      )
      # Repeat terminal one-step across critic history.
      for i in range(self.critic_history_length):
        sl = slice(
          i * self.num_one_step_privileged_obs,
          (i + 1) * self.num_one_step_privileged_obs,
        )
        term_hist[:, sl] = term_priv
    else:
      term_hist = torch.zeros(
        0, self.num_privileged_obs, device=self.device
      )

    return (
      self.get_observations(),
      self.get_privileged_observations(),
      rew,
      dones,
      extras,
      term_ids,
      term_hist,
    )

  @property
  def unwrapped(self):
    return self.env.unwrapped

  @property
  def max_episode_length(self):
    return self.env.max_episode_length

  @property
  def episode_length_buf(self):
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value):
    self.env.episode_length_buf = value
