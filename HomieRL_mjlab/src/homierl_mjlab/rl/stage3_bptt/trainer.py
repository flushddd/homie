"""World-model fit + critic TD + short-horizon BPTT actor updates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor, Stage3Critic, WorldModel
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import stage3_reward


class Stage3BpttTrainer:
  """Alternating: WM → critic → BPTT actor (SuperMimic-style short AC).

  Imagination control synthesis (v1):
    - ``control = stitch(logged_loco_u, dls(arm | delta_p))`` is provided by
      ``control_fn(state, delta_p, goal, geom) -> control``.
    - Palm / clearance for rewards via ``geom_fn(state, goal, geom)``.
  """

  def __init__(
    self,
    cfg: Stage3BpttCfg,
    *,
    state_dim: int,
    control_dim: int,
    obs_dim: int,
    feat_dim: int,
    goal_dim: int,
    geom_dim: int,
    control_fn: Callable[..., torch.Tensor],
    obs_fn: Callable[..., torch.Tensor],
    feat_fn: Callable[..., torch.Tensor],
    geom_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    num_envs: int = 1,
  ):
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.control_fn = control_fn
    self.obs_fn = obs_fn
    self.feat_fn = feat_fn
    self.geom_fn = geom_fn
    self.num_envs_hint = int(num_envs)

    self.actor = Stage3Actor(obs_dim, cfg.actor_hidden, cfg.action_scale_m).to(self.device)
    self.critic = Stage3Critic(feat_dim, cfg.critic_hidden).to(self.device)
    self.critic_tgt = Stage3Critic(feat_dim, cfg.critic_hidden).to(self.device)
    self.critic_tgt.load_state_dict(self.critic.state_dict())
    self.wm = WorldModel(
      state_dim, control_dim, ctx_dim=goal_dim + geom_dim, hidden=cfg.wm_hidden
    ).to(self.device)

    self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
    self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.wm_lr)

    self.replay = EpisodeReplay(
      cfg.replay_capacity,
      state_dim,
      control_dim,
      goal_dim,
      geom_dim,
      self.device,
      num_envs=self.num_envs_hint,
    )
    self.last_action = torch.zeros(1, 3, device=self.device)
    self.iter = 0

  def update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    states, controls, _, _, goals, geoms = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=getattr(self, "num_envs_hint", None)
    )
    # states (B,H+1,S), controls (B,H,U)
    ctx = torch.cat((goals, geoms), dim=-1)
    pred = []
    s = states[:, 0]
    for t in range(cfg.wm_window):
      s = self.wm(s, controls[:, t], ctx)
      pred.append(s)
    pred_s = torch.stack(pred, dim=1)
    target = states[:, 1:]
    loss = F.mse_loss(pred_s, target)
    self.opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.wm.parameters(), cfg.grad_clip)
    self.opt_wm.step()
    return {"wm_loss": float(loss.item())}

  def update_critic(self) -> dict[str, float]:
    cfg = self.cfg
    states, controls, rewards, terms, goals, geoms = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=getattr(self, "num_envs_hint", None)
    )
    b, h, _ = controls.shape
    # TD(λ) on real windows; bootstrap with target critic on non-terminals.
    feats = []
    for t in range(h + 1):
      feats.append(self.feat_fn(states[:, t], goals, geoms))
    # V_t
    values = torch.stack([self.critic(f) for f in feats], dim=1)  # (B,H+1)
    with torch.no_grad():
      v_tgt = torch.stack([self.critic_tgt(f) for f in feats], dim=1)
      returns = torch.zeros(b, h, device=self.device)
      gae = torch.zeros(b, device=self.device)
      for t in reversed(range(h)):
        mask = 1.0 - terms[:, t]
        # truncation treated as bootstrap (env-defined); true term → no bootstrap
        delta = rewards[:, t] + cfg.gamma * mask * v_tgt[:, t + 1] - v_tgt[:, t]
        gae = delta + cfg.gamma * cfg.lam * mask * gae
        returns[:, t] = gae + v_tgt[:, t]
    loss = F.mse_loss(values[:, :-1], returns.detach())
    self.opt_critic.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
    self.opt_critic.step()
    # soft update target
    with torch.no_grad():
      for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
        pt.data.mul_(1.0 - cfg.target_critic_tau).add_(cfg.target_critic_tau * p.data)
    return {"critic_loss": float(loss.item())}

  def update_actor_bptt(self) -> dict[str, float]:
    """Freeze WM + critic weights; BPTT through imagined H steps."""
    cfg = self.cfg
    states, _, _, _, goals, geoms = self.replay.sample_contiguous_windows(
      max(32, cfg.wm_batch // 4),
      1,
      num_envs_hint=getattr(self, "num_envs_hint", None),
    )
    # start from s0 only
    s = states[:, 0].detach().requires_grad_(True)
    ctx = torch.cat((goals, geoms), dim=-1)
    last_a = torch.zeros(s.shape[0], 3, device=self.device)
    returns = []
    discount = torch.ones(s.shape[0], device=self.device)
    surv = torch.ones(s.shape[0], device=self.device)

    # Freeze modules' parameters but keep graph through inputs.
    for p in self.wm.parameters():
      p.requires_grad_(False)
    for p in self.critic.parameters():
      p.requires_grad_(False)

    for t in range(cfg.imagine_horizon):
      obs = self.obs_fn(s, goals, geoms, last_a)
      a = self.actor(obs)
      u = self.control_fn(s, a, goals, geoms)
      s = self.wm(s, u, ctx)
      palm, clearance = self.geom_fn(s, goals, geoms)
      r = stage3_reward(
        palm_pos=palm,
        goal_pos=goals[..., :3],
        clearance=clearance,
        action=a,
        last_action=last_a,
        sigma_p=cfg.sigma_p,
        d_safe=cfg.d_safe,
        d_max=cfg.d_max,
        w_target=cfg.w_target,
        w_safe=cfg.w_safe,
        w_smooth=cfg.w_smooth,
        w_stability=cfg.w_stability,
      )
      # Hard-mask continuation unused in v1 imagination (SDF risk in reward).
      returns.append(discount * surv * r)
      discount = discount * cfg.gamma
      last_a = a

    # Terminal value — keep input grad, frozen params.
    feat_h = self.feat_fn(s, goals, geoms)
    v_h = self.critic(feat_h)
    j = torch.stack(returns, dim=0).sum(0) + discount * surv * v_h
    loss = -j.mean()

    self.opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip))
    self.opt_actor.step()

    for p in self.wm.parameters():
      p.requires_grad_(True)
    for p in self.critic.parameters():
      p.requires_grad_(True)

    return {
      "actor_loss": float(loss.item()),
      "actor_grad_norm": gn,
      "J_mean": float(j.detach().mean()),
    }

  def train_iteration(self) -> dict[str, float]:
    logs: dict[str, float] = {}
    for _ in range(self.cfg.wm_updates_per_iter):
      logs.update(self.update_world_model())
    for _ in range(self.cfg.critic_updates_per_iter):
      logs.update(self.update_critic())
    for _ in range(self.cfg.actor_updates_per_iter):
      logs.update(self.update_actor_bptt())
    self.iter += 1
    logs["iter"] = float(self.iter)
    return logs

  def state_dict(self) -> dict:
    return {
      "actor": self.actor.state_dict(),
      "critic": self.critic.state_dict(),
      "critic_tgt": self.critic_tgt.state_dict(),
      "wm": self.wm.state_dict(),
      "opt_actor": self.opt_actor.state_dict(),
      "opt_critic": self.opt_critic.state_dict(),
      "opt_wm": self.opt_wm.state_dict(),
      "iter": self.iter,
      "cfg": asdict(self.cfg),
    }

  def load_state_dict(self, sd: dict) -> None:
    self.actor.load_state_dict(sd["actor"])
    self.critic.load_state_dict(sd["critic"])
    self.critic_tgt.load_state_dict(sd["critic_tgt"])
    self.wm.load_state_dict(sd["wm"])
    self.opt_actor.load_state_dict(sd["opt_actor"])
    self.opt_critic.load_state_dict(sd["opt_critic"])
    self.opt_wm.load_state_dict(sd["opt_wm"])
    self.iter = int(sd.get("iter", 0))
