"""Cleaned HIMPPO with left-right symmetry for Homie G1 (12 leg actions)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic
from homierl_mjlab.rl.him.him_rollout_storage import HIMRolloutStorage


class HIMPPO:
  actor_critic: HIMActorCritic

  def __init__(
    self,
    actor_critic: HIMActorCritic,
    use_flip: bool = True,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    device: str = "cpu",
    symmetry_scale: float = 1.0,  # Homie G1_*CfgPPO.algorithm.symmetry_scale
    num_dof: int = 29,
    max_learning_rate: float = 1e-2,  # Homie adaptive clamp
    min_learning_rate: float = 1e-5,
  ):
    self.device = device
    self.use_flip = use_flip
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.max_learning_rate = max_learning_rate
    self.min_learning_rate = min_learning_rate
    self.symmetry_scale = symmetry_scale
    self.num_dof = num_dof

    self.actor_critic = actor_critic
    self.actor_critic.to(self.device)
    self.storage = None
    self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    self.transition = HIMRolloutStorage.Transition()
    self.transition_sym = HIMRolloutStorage.Transition()

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    # Build left-right joint permutation for G1 naming convention.
    self._joint_perm: list[int] | None = None
    self._joint_sign: list[float] | None = None

  def set_joint_symmetry(self, joint_names: list[str]) -> None:
    """Configure obs joint flip from ordered joint names."""
    n = len(joint_names)
    name_to_idx = {n_: i for i, n_ in enumerate(joint_names)}
    perm = list(range(n))
    sign = [1.0] * n
    for i, name in enumerate(joint_names):
      if name.startswith("left_"):
        right = "right_" + name[len("left_") :]
        if right in name_to_idx:
          perm[i] = name_to_idx[right]
      elif name.startswith("right_"):
        left = "left_" + name[len("right_") :]
        if left in name_to_idx:
          perm[i] = name_to_idx[left]
      # Mirror-sensitive axes (roll / yaw) get a sign flip.
      if any(k in name for k in ("roll", "yaw")) and (
        name.startswith("left_") or name.startswith("right_") or name.startswith("waist_")
      ):
        sign[i] = -1.0
    self._joint_perm = perm
    self._joint_sign = sign
    self.num_dof = n

  def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
    self.storage = HIMRolloutStorage(
      num_envs,
      num_transitions_per_env,
      actor_obs_shape,
      critic_obs_shape,
      action_shape,
      self.device,
    )

  def test_mode(self):
    self.actor_critic.test()

  def train_mode(self):
    self.actor_critic.train()

  def act(self, obs, critic_obs):
    self.transition.actions = self.actor_critic.act(obs).detach()
    self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
    self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.actor_critic.action_mean.detach()
    self.transition.action_sigma = self.actor_critic.action_std.detach()
    self.transition.observations = obs
    self.transition.critic_observations = critic_obs

    if self.use_flip:
      obs_sym = self.flip_actor_obs(obs)
      critic_sym = self.flip_critic_obs(critic_obs)
      self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
      self.transition_sym.values = self.actor_critic.evaluate(critic_sym).detach()
      self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(
        self.transition_sym.actions
      ).detach()
      self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
      self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
      self.transition_sym.observations = obs_sym
      self.transition_sym.critic_observations = critic_sym

    return self.transition.actions

  def process_env_step(self, rewards, dones, infos, next_critic_obs):
    self.transition.next_critic_observations = next_critic_obs.clone()
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones

    if self.use_flip:
      next_sym = self.flip_critic_obs(next_critic_obs)
      self.transition_sym.next_critic_observations = next_sym.clone()
      self.transition_sym.rewards = rewards.clone()
      self.transition_sym.dones = dones

    if "time_outs" in infos:
      timeouts = infos["time_outs"].unsqueeze(1).to(self.device)
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * timeouts, 1
      )
      if self.use_flip:
        self.transition_sym.rewards += self.gamma * torch.squeeze(
          self.transition_sym.values * timeouts, 1
        )

    self.storage.add_transitions(self.transition)
    if self.use_flip:
      self.storage.add_transitions(self.transition_sym)
      self.transition_sym.clear()
    self.transition.clear()
    self.actor_critic.reset(dones)

  def compute_returns(self, last_critic_obs):
    last_values = self.actor_critic.evaluate(last_critic_obs).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def update(self):
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_estimation_loss = 0.0
    mean_swap_loss = 0.0
    mean_actor_sym_loss = 0.0
    mean_critic_sym_loss = 0.0
    mean_loss_keep = 0.0

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for (
      obs_batch,
      critic_obs_batch,
      actions_batch,
      next_critic_obs_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
    ) in generator:
      self.actor_critic.act(obs_batch)
      actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
      value_batch = self.actor_critic.evaluate(critic_obs_batch)
      mu_batch = self.actor_critic.action_mean
      sigma_batch = self.actor_critic.action_std
      entropy_batch = self.actor_critic.entropy

      # Adaptive LR via KL.
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            dim=-1,
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(
              self.min_learning_rate, self.learning_rate / 1.5
            )
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(
              self.max_learning_rate, self.learning_rate * 1.5
            )
          for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      # Symmetry consistency loss.
      actor_sym_loss = torch.zeros((), device=self.device)
      critic_sym_loss = torch.zeros((), device=self.device)
      if self.use_flip:
        flipped_obs = self.flip_actor_obs(obs_batch)
        flipped_critic = self.flip_critic_obs(critic_obs_batch)
        pred = self.actor_critic.act_inference(obs_batch)
        pred_flip = self.actor_critic.act_inference(flipped_obs)
        actor_sym_loss = self.symmetry_scale * torch.mean(
          torch.sum(torch.square(pred_flip - self.flip_actions(pred)), dim=-1)
        )
        v = self.actor_critic.evaluate(critic_obs_batch)
        v_f = self.actor_critic.evaluate(flipped_critic)
        critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(v_f - v.detach()))

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + actor_sym_loss
        + critic_sym_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      estimation_loss, swap_loss = self.actor_critic.update_estimator(
        obs_batch, next_critic_obs_batch, lr=self.learning_rate
      )

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_estimation_loss += estimation_loss
      mean_swap_loss += swap_loss
      mean_actor_sym_loss += float(actor_sym_loss.detach())
      mean_critic_sym_loss += float(critic_sym_loss.detach())

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return (
      mean_value_loss / num_updates,
      mean_surrogate_loss / num_updates,
      mean_estimation_loss / num_updates,
      mean_swap_loss / num_updates,
      mean_actor_sym_loss / num_updates,
      mean_critic_sym_loss / num_updates,
      mean_loss_keep,
    )

  # ---- symmetry helpers (Homie obs layout) ----
  # one-step: [vx,vy,yaw, height, ang(3), grav(3), q(N), dq(N), a(12)]

  def _flip_one_step(self, step: torch.Tensor, with_lin_vel: bool) -> torch.Tensor:
    out = step.clone()
    n = self.num_dof
    # commands
    out[..., 1] = -step[..., 1]
    out[..., 2] = -step[..., 2]
    # ang vel
    out[..., 4] = -step[..., 4]
    out[..., 6] = -step[..., 6]
    # gravity
    out[..., 8] = -step[..., 8]
    # joints
    q0 = 10
    if self._joint_perm is not None:
      perm = self._joint_perm
      sign = self._joint_sign
      for i in range(n):
        out[..., q0 + i] = sign[i] * step[..., q0 + perm[i]]
        out[..., q0 + n + i] = sign[i] * step[..., q0 + n + perm[i]]
    # actions (12 legs) — assume [L6 | R6] pitch,roll,yaw,knee,ap,ar
    a0 = 10 + 2 * n
    if step.shape[-1] >= a0 + 12:
      out[..., a0 : a0 + 12] = self.flip_actions(step[..., a0 : a0 + 12])
    if with_lin_vel and step.shape[-1] >= a0 + 12 + 3:
      lv = a0 + 12
      out[..., lv + 1] = -step[..., lv + 1]
    return out

  def flip_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
    h = self.actor_critic.actor_history_length
    d = self.actor_critic.num_one_step_obs
    x = obs[:, : h * d].view(-1, h, d)
    y = self._flip_one_step(x, with_lin_vel=False)
    return y.reshape(obs.shape[0], h * d)

  def flip_critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
    h = self.actor_critic.critic_history_length
    d = self.actor_critic.num_one_step_critic_obs
    x = obs[:, : h * d].view(-1, h, d)
    y = self._flip_one_step(x, with_lin_vel=True)
    return y.reshape(obs.shape[0], h * d)

  def flip_actions(self, actions: torch.Tensor) -> torch.Tensor:
    """Mirror 12 leg actions along the last dimension."""
    flat = actions.reshape(-1, actions.shape[-1])
    out = torch.zeros_like(flat)
    out[:, 0] = flat[:, 6]
    out[:, 1] = -flat[:, 7]
    out[:, 2] = -flat[:, 8]
    out[:, 3] = flat[:, 9]
    out[:, 4] = flat[:, 10]
    out[:, 5] = -flat[:, 11]
    out[:, 6] = flat[:, 0]
    out[:, 7] = -flat[:, 1]
    out[:, 8] = -flat[:, 2]
    out[:, 9] = flat[:, 3]
    out[:, 10] = flat[:, 4]
    out[:, 11] = -flat[:, 5]
    return out.reshape(actions.shape)
