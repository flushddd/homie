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

    # Teacher mean+KL lock (phase-2: 12 legs; phase-3: 14 loco, no waist yaw).
    self.teacher_actor_critic: HIMActorCritic | None = None
    self.keep_loco_weight = 0.0
    self.keep_loco_dims = 12
    self.freeze_std_dims = 12
    self.stand_height = 0.74
    self.keep_height_tol = 0.02
    self.is_waist_one_step_idx = 86  # unified: 68+17+1
    self.keep_loco_always = False
    self.teacher_one_step_obs = 0

    # Homie flip_g1: legs+waist only; arms stay 0 in flipped obs.
    self._joint_names: list[str] = []

  def set_joint_symmetry(self, joint_names: list[str]) -> None:
    """Bind robot joint names; flip follows Homie ``flip_g1_*`` (arms never filled)."""
    self._joint_names = list(joint_names)
    self.num_dof = len(joint_names)
    n_arm = sum(
      1
      for n in joint_names
      if any(k in n for k in ("shoulder", "elbow", "wrist"))
    )
    print(
      f"[INFO] Joint symmetry (Homie flip_g1): ndof={self.num_dof}, "
      f"arms_never_flipped={n_arm}"
    )

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
    # Guard against reward spikes / NaNs that later poison value targets.
    self.transition.rewards = torch.nan_to_num(
      rewards, nan=0.0, posinf=100.0, neginf=-100.0
    ).clamp(-100.0, 100.0)
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

      # ---- Teacher mean + KL keep on loco dims (selective via is_waist mask) ----
      loss_keep = torch.zeros((), device=self.device)
      if self.teacher_actor_critic is not None and self.keep_loco_weight > 0:
        with torch.no_grad():
          obs_t = self._teacher_obs(obs_batch)
          # Inference only (no Normal sample — avoids CUDA std assert on teacher).
          teacher_mu = self.teacher_actor_critic.act_inference(obs_t)
          teacher_sigma = (
            torch.nan_to_num(self.teacher_actor_critic.std, nan=1.0)
            .abs()
            .clamp(min=1e-4)
            .expand_as(teacher_mu)
          )
        k = min(int(self.keep_loco_dims), mu_batch.shape[-1], teacher_mu.shape[-1])
        loco_mask = self._loco_mask(obs_batch)
        mask_count = loco_mask.sum().clamp(min=1.0)
        diff = (mu_batch[:, :k] - teacher_mu[:, :k]) * loco_mask.unsqueeze(-1)
        # Homie ``_kl_diag_gauss`` detaches student → KL has no grad; mean L2
        # carries the keep signal (selective on non-waist envs / always in p3).
        kl = self._kl_diag_gauss(
          mu_batch[:, :k],
          sigma_batch[:, :k],
          teacher_mu[:, :k],
          teacher_sigma[:, :k],
          detach_student=True,
        ) * loco_mask
        loss_keep = (kl.sum() + (diff * diff).sum()) / (mask_count * float(k))

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

      # Estimator update before policy loss (Homie order).
      estimation_loss, swap_loss = self.actor_critic.update_estimator(
        obs_batch, next_critic_obs_batch, lr=self.learning_rate
      )

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
        + self.keep_loco_weight * loss_keep
      )

      self.optimizer.zero_grad()
      loss.backward()
      # Freeze exploration noise on locked loco dims (Homie freeze_std).
      if (
        self.teacher_actor_critic is not None
        and hasattr(self.actor_critic, "std")
        and self.actor_critic.std.grad is not None
        and self.freeze_std_dims > 0
      ):
        self.actor_critic.std.grad[: self.freeze_std_dims].zero_()
      nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      # Keep Diagonal Normal valid after a bad minibatch (EE dims not freeze_std).
      if hasattr(self.actor_critic, "std"):
        with torch.no_grad():
          self.actor_critic.std.copy_(
            torch.nan_to_num(self.actor_critic.std, nan=1.0, posinf=1.0, neginf=1.0)
            .abs()
            .clamp(min=1e-4, max=2.0)
          )

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_estimation_loss += estimation_loss
      mean_swap_loss += swap_loss
      mean_actor_sym_loss += float(actor_sym_loss.detach())
      mean_critic_sym_loss += float(critic_sym_loss.detach())
      mean_loss_keep += float(loss_keep.detach())

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return (
      mean_value_loss / num_updates,
      mean_surrogate_loss / num_updates,
      mean_estimation_loss / num_updates,
      mean_swap_loss / num_updates,
      mean_actor_sym_loss / num_updates,
      mean_critic_sym_loss / num_updates,
      mean_loss_keep / num_updates,
    )

  # ---- symmetry helpers (Homie ``flip_g1_*``) ----
  # G1 hinge order: L6 | R6 | waist3 | arms14. Homie fills ONLY legs+waist
  # (+ loco actions); arms / grasp / SDF stay 0 via zeros_like.
  _ACT_OBS = 17
  _GRASP = 7
  _N_DOF = 29
  _Q0 = 10
  _A0 = 10 + 2 * 29  # 68

  def _flip_one_step(self, step: torch.Tensor, with_lin_vel: bool) -> torch.Tensor:
    """Port of Homie ``flip_g1_actor_obs`` / ``flip_g1_critic_obs``.

    ``zeros_like`` then write only the loco channels Homie writes. Arms
    (idx 15..28) and grasp/SDF/clearance/EE trail are never written → stay 0.
    """
    out = torch.zeros_like(step)
    # commands
    out[..., 0] = step[..., 0]
    out[..., 1] = -step[..., 1]
    out[..., 2] = -step[..., 2]
    out[..., 3] = step[..., 3]
    # ang vel
    out[..., 4] = -step[..., 4]
    out[..., 5] = step[..., 5]
    out[..., 6] = -step[..., 6]
    # gravity
    out[..., 7] = step[..., 7]
    out[..., 8] = -step[..., 8]
    out[..., 9] = step[..., 9]

    q0 = self._Q0
    # joint pos: legs L↔R, waist signs (arms left at 0)
    out[..., q0 + 0] = step[..., q0 + 6]
    out[..., q0 + 1] = -step[..., q0 + 7]
    out[..., q0 + 2] = -step[..., q0 + 8]
    out[..., q0 + 3] = step[..., q0 + 9]
    out[..., q0 + 4] = step[..., q0 + 10]
    out[..., q0 + 5] = -step[..., q0 + 11]
    out[..., q0 + 6] = step[..., q0 + 0]
    out[..., q0 + 7] = -step[..., q0 + 1]
    out[..., q0 + 8] = -step[..., q0 + 2]
    out[..., q0 + 9] = step[..., q0 + 3]
    out[..., q0 + 10] = step[..., q0 + 4]
    out[..., q0 + 11] = -step[..., q0 + 5]
    out[..., q0 + 12] = -step[..., q0 + 12]
    out[..., q0 + 13] = -step[..., q0 + 13]
    out[..., q0 + 14] = step[..., q0 + 14]

    v0 = q0 + self._N_DOF
    out[..., v0 + 0] = step[..., v0 + 6]
    out[..., v0 + 1] = -step[..., v0 + 7]
    out[..., v0 + 2] = -step[..., v0 + 8]
    out[..., v0 + 3] = step[..., v0 + 9]
    out[..., v0 + 4] = step[..., v0 + 10]
    out[..., v0 + 5] = -step[..., v0 + 11]
    out[..., v0 + 6] = step[..., v0 + 0]
    out[..., v0 + 7] = -step[..., v0 + 1]
    out[..., v0 + 8] = -step[..., v0 + 2]
    out[..., v0 + 9] = step[..., v0 + 3]
    out[..., v0 + 10] = step[..., v0 + 4]
    out[..., v0 + 11] = -step[..., v0 + 5]
    out[..., v0 + 12] = -step[..., v0 + 12]
    out[..., v0 + 13] = -step[..., v0 + 13]
    out[..., v0 + 14] = step[..., v0 + 14]

    # last actions: Homie fills 12 legs; we also flip waist/EE if present
    a0 = self._A0
    n_act = int(self.actor_critic.num_actions)
    if step.shape[-1] >= a0 + n_act:
      out[..., a0 : a0 + n_act] = self.flip_actions(step[..., a0 : a0 + n_act])

    if with_lin_vel and step.shape[-1] >= 3:
      lv = int(step.shape[-1]) - 3
      out[..., lv] = step[..., lv]
      out[..., lv + 1] = -step[..., lv + 1]
      out[..., lv + 2] = step[..., lv + 2]
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
    """Homie ``flip_g1_actions`` (12 legs) + waist/EE for later phases."""
    flat = actions.reshape(-1, actions.shape[-1])
    out = torch.zeros_like(flat)
    # phase-1: exactly Homie 12-leg L6|R6
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
    if flat.shape[-1] >= 14:
      out[:, 12] = -flat[:, 12]  # waist_roll
      out[:, 13] = flat[:, 13]  # waist_pitch
      if flat.shape[-1] >= 17:
        out[:, 14] = flat[:, 14]
        out[:, 15] = -flat[:, 15]
        out[:, 16] = flat[:, 16]
        if flat.shape[-1] > 17:
          out[:, 17:] = flat[:, 17:]
    return out.reshape(actions.shape)

  def set_teacher(
    self,
    teacher_actor_critic: HIMActorCritic,
    keep_loco_weight: float = 2.0,
    keep_loco_dims: int = 12,
    freeze_std_dims: int = 12,
    stand_height: float = 0.74,
    keep_height_tol: float = 0.02,
    is_waist_one_step_idx: int = 86,
    keep_loco_always: bool = False,
    teacher_one_step_obs: int = 0,
  ) -> None:
    """Freeze teacher AC and enable mean+KL loco lock (Homie ``set_teacher``)."""
    self.teacher_actor_critic = teacher_actor_critic
    self.teacher_actor_critic.to(self.device)
    self.teacher_actor_critic.eval()
    for p in self.teacher_actor_critic.parameters():
      p.requires_grad = False
    self.keep_loco_weight = float(keep_loco_weight)
    self.keep_loco_dims = int(keep_loco_dims)
    self.freeze_std_dims = int(freeze_std_dims)
    self.stand_height = float(stand_height)
    self.keep_height_tol = float(keep_height_tol)
    self.is_waist_one_step_idx = int(is_waist_one_step_idx)
    self.keep_loco_always = bool(keep_loco_always)
    self.teacher_one_step_obs = int(teacher_one_step_obs)
    print(
      f"[INFO] Teacher lock ON: weight={self.keep_loco_weight}, "
      f"dims={self.keep_loco_dims}, freeze_std={self.freeze_std_dims}, "
      f"is_waist_idx={self.is_waist_one_step_idx}, "
      f"always={self.keep_loco_always}, teacher_d={self.teacher_one_step_obs}"
    )

  def _teacher_obs(self, obs_batch: torch.Tensor) -> torch.Tensor:
    """Zero channels the frozen teacher never trained on.

    Shared layout: proprio68 | a17 | grasp7 | sdf128 | clearance6 | ee_err3.
    Phase-2 teacher (p1): loco a12 only; grasp/SDF stay 0.
    Phase-3 teacher (p2): loco a14 + grasp; SDF/EE-act stay 0.
    """
    if self.teacher_actor_critic is None:
      return obs_batch
    d = int(self.actor_critic.num_one_step_obs)
    h = int(self.actor_critic.actor_history_length)
    proprio = 68
    act_slots = self._ACT_OBS if d >= proprio + self._ACT_OBS + self._GRASP else int(
      self.actor_critic.num_actions
    )
    a_t = min(int(self.keep_loco_dims), act_slots)
    # p3 keep_loco_dims=14 → teacher saw grasp; p2 keep=12 → teacher did not.
    copy_grasp = int(self.keep_loco_dims) >= 14
    grasp_dim = self._GRASP
    obs_t = torch.zeros_like(obs_batch)
    for t in range(h):
      o0 = t * d
      src = obs_batch[:, o0 : o0 + d]
      frame = obs_t[:, o0 : o0 + d]
      frame[:, :proprio] = src[:, :proprio]
      frame[:, proprio : proprio + a_t] = src[:, proprio : proprio + a_t]
      if copy_grasp:
        grasp0 = proprio + act_slots
        if src.shape[-1] >= grasp0 + grasp_dim:
          frame[:, grasp0 : grasp0 + grasp_dim] = src[:, grasp0 : grasp0 + grasp_dim]
    return torch.nan_to_num(obs_t, nan=0.0, posinf=0.0, neginf=0.0)

  # Unified one-step: proprio68|a17|grasp7|sdf… → is_waist @ 86

  def _loco_mask(self, obs_batch: torch.Tensor) -> torch.Tensor:
    """Mask for loco keep. Phase-2: only non-waist envs. Phase-3: always on."""
    if self.keep_loco_always:
      return torch.ones(obs_batch.shape[0], device=obs_batch.device, dtype=obs_batch.dtype)
    d = int(self.actor_critic.num_one_step_obs)
    h = int(self.actor_critic.actor_history_length)
    base = (h - 1) * d
    idx = int(self.is_waist_one_step_idx)
    if d > idx:
      is_waist = obs_batch[:, base + idx]
      return (is_waist < 0.5).float()
    height = obs_batch[:, base + 3]
    return (torch.abs(height - self.stand_height) < self.keep_height_tol).float()

  @staticmethod
  def _kl_diag_gauss(
    mu_s: torch.Tensor,
    sig_s: torch.Tensor,
    mu_t: torch.Tensor,
    sig_t: torch.Tensor,
    eps: float = 1e-8,
    detach_student: bool = False,
  ) -> torch.Tensor:
    """KL(N(mu_s,sig_s^2) || N(mu_t,sig_t^2)) for diagonal Gaussians, shape (B,)."""
    if detach_student:
      mu_s = mu_s.detach()
      sig_s = sig_s.detach()
    mu_t = mu_t.detach()
    sig_t = sig_t.detach()
    var_s = sig_s * sig_s
    var_t = sig_t * sig_t
    kl = 0.5 * (
      (var_s + (mu_s - mu_t).pow(2)) / (var_t + eps)
      - 1.0
      + 2.0 * torch.log((sig_t + eps) / (sig_s + eps))
    )
    return kl.sum(dim=-1)
