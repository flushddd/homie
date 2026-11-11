"""WM + critic + BPTT for Stage-1/2 loco SuperMimic experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
from homierl_mjlab.rl.stage12_bptt.networks import LocoActor, LocoCritic, WorldModel
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import (
  soft_stability_reward,
  tilt_angle_from_root_quat,
)
from homierl_mjlab.rl.stage3_bptt.trainer import _finite, _params_finite, _sanitize, math_isfinite


class Stage12BpttTrainer:
  def __init__(
    self,
    cfg: Stage12BpttCfg,
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
    cfg.resolve_dims()
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.control_fn = control_fn
    self.obs_fn = obs_fn
    self.feat_fn = feat_fn
    self.geom_fn = geom_fn
    self.num_envs_hint = int(num_envs)
    self.body_dim = int(cfg.body_dim)
    self.action_semantics = str(cfg.action_semantics)
    self.n_joints_hint = int(control_dim)

    self.actor = LocoActor(
      obs_dim, body_dim=self.body_dim, hidden=cfg.actor_hidden, near_zero_init=True
    ).to(self.device)
    self.critic = LocoCritic(feat_dim, cfg.critic_hidden).to(self.device)
    self.critic_tgt = LocoCritic(feat_dim, cfg.critic_hidden).to(self.device)
    self.critic_tgt.load_state_dict(self.critic.state_dict())
    self.wm = WorldModel(
      state_dim,
      control_dim,
      ctx_dim=goal_dim + geom_dim,
      hidden=cfg.wm_hidden,
      residual_clip=float(cfg.wm_residual_clip),
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
      actor_action_dim=self.body_dim,
      teacher_body_dim=self.body_dim if float(cfg.w_teacher) > 0 else 0,
      action_semantics=self.action_semantics,
    )
    self.iter = 0
    self.nan_streak = 0
    self._last_good: dict | None = None
    self.snapshot_good()

  def snapshot_good(self) -> None:
    self._last_good = {
      "actor": deepcopy(self.actor.state_dict()),
      "critic": deepcopy(self.critic.state_dict()),
      "critic_tgt": deepcopy(self.critic_tgt.state_dict()),
      "wm": deepcopy(self.wm.state_dict()),
    }

  def restore_good(self) -> bool:
    if self._last_good is None:
      return False
    self.actor.load_state_dict(self._last_good["actor"])
    self.critic.load_state_dict(self._last_good["critic"])
    self.critic_tgt.load_state_dict(self._last_good["critic_tgt"])
    self.wm.load_state_dict(self._last_good["wm"])
    return True

  def decay_lrs(self, factor: float) -> None:
    f = float(factor)
    for opt in (self.opt_wm, self.opt_critic, self.opt_actor):
      for pg in opt.param_groups:
        pg["lr"] = float(pg["lr"]) * f

  def _step_or_skip(
    self,
    *,
    loss: torch.Tensor,
    opt: torch.optim.Optimizer,
    module: nn.Module,
    key: str,
  ) -> dict[str, float]:
    cfg = self.cfg
    if not _finite(loss):
      opt.zero_grad(set_to_none=True)
      return {key: float("nan"), f"{key}_skipped": 1.0}
    before = deepcopy(module.state_dict()) if cfg.nan_guard else None
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip))
    if cfg.nan_guard and (not math_isfinite(gn)):
      opt.zero_grad(set_to_none=True)
      return {key: float(loss.detach().item()), f"{key}_skipped": 1.0, f"{key}_grad_norm": gn}
    opt.step()
    if cfg.nan_guard and not _params_finite(module):
      assert before is not None
      module.load_state_dict(before)
      opt.zero_grad(set_to_none=True)
      return {key: float("nan"), f"{key}_restored": 1.0}
    return {key: float(loss.detach().item()), f"{key}_grad_norm": gn}

  def _unpack_windows(self, windows: tuple) -> tuple:
    if len(windows) == 10:
      return windows
    if len(windows) == 8:
      return windows + (None, None)
    raise RuntimeError(f"Unexpected window tuple len={len(windows)}")

  def _tilt_from_state(self, state: torch.Tensor) -> torch.Tensor:
    # state = q|qd|pos3|quat4|lin3|ang3
    nj = (state.shape[-1] - 13) // 2
    quat = state[..., 2 * nj + 3 : 2 * nj + 7]
    return tilt_angle_from_root_quat(quat)

  def update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    windows = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=self.num_envs_hint
    )
    states, controls, _, _, _, goals, geoms, *_ = self._unpack_windows(windows)
    states, controls, goals, geoms = _sanitize(states, controls, goals, geoms)
    ctx = torch.cat((goals, geoms), dim=-1)
    pred = []
    s = states[:, 0]
    for t in range(cfg.wm_window):
      s = self.wm(s, controls[:, t], ctx)
      pred.append(s)
    pred_s = torch.stack(pred, dim=1)
    loss = F.huber_loss(pred_s, states[:, 1:], delta=float(cfg.wm_huber_delta))
    return self._step_or_skip(loss=loss, opt=self.opt_wm, module=self.wm, key="wm_loss")

  def update_critic(self) -> dict[str, float]:
    cfg = self.cfg
    windows = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=self.num_envs_hint
    )
    (
      states,
      controls,
      rewards,
      terms,
      _truncs,
      goals,
      geoms,
      _present,
      _teacher,
      _actor_a,
    ) = self._unpack_windows(windows)
    states, controls, rewards, terms, goals, geoms = _sanitize(
      states, controls, rewards, terms, goals, geoms
    )
    clip = float(cfg.reward_clip)
    clipped = rewards.clamp(-clip, clip) if cfg.clip_critic_reward else rewards
    b, h, _ = controls.shape
    feats = [self.feat_fn(states[:, t], goals, geoms) for t in range(h + 1)]
    values = torch.stack([self.critic(f) for f in feats], dim=1)
    with torch.no_grad():
      v_tgt = torch.stack([self.critic_tgt(f) for f in feats], dim=1)
      returns = torch.zeros(b, h, device=self.device)
      gae = torch.zeros(b, device=self.device)
      for t in reversed(range(h)):
        mask = 1.0 - terms[:, t]
        delta = clipped[:, t] + cfg.gamma * mask * v_tgt[:, t + 1] - v_tgt[:, t]
        gae = delta + cfg.gamma * cfg.lam * mask * gae
        returns[:, t] = gae + v_tgt[:, t]
    loss = F.mse_loss(values[:, :-1], returns.detach())
    out = self._step_or_skip(
      loss=loss, opt=self.opt_critic, module=self.critic, key="critic_loss"
    )
    if out.get("critic_loss_skipped") or out.get("critic_loss_restored"):
      return out
    with torch.no_grad():
      for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
        pt.data.mul_(1.0 - cfg.target_critic_tau).add_(cfg.target_critic_tau * p.data)
    return out

  def update_actor_bptt(self) -> dict[str, float]:
    cfg = self.cfg
    windows = self.replay.sample_contiguous_windows(
      max(32, cfg.wm_batch // 4), 1, num_envs_hint=self.num_envs_hint
    )
    (
      states,
      _controls,
      _rewards,
      _terms,
      _truncs,
      goals,
      geoms,
      _present,
      teacher_body,
      _actor_logged,
    ) = self._unpack_windows(windows)
    states, goals, geoms = _sanitize(states, goals, geoms)
    s = states[:, 0].detach().requires_grad_(True)
    ctx = torch.cat((goals, geoms), dim=-1)
    last_a = torch.zeros(s.shape[0], self.body_dim, device=self.device)
    returns = []
    discount = torch.ones(s.shape[0], device=self.device)
    surv = torch.ones(s.shape[0], device=self.device)
    clip = float(cfg.reward_clip)
    teacher_mse = None
    a0 = None

    for p in self.wm.parameters():
      p.requires_grad_(False)
    for p in self.critic.parameters():
      p.requires_grad_(False)

    for _t in range(cfg.imagine_horizon):
      obs = self.obs_fn(s, goals, geoms, last_a)
      obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
      a = self.actor(obs)
      if a0 is None:
        a0 = a
      u = self.control_fn(s, a, goals, geoms)
      s = self.wm(s, u, ctx)
      tilt = self._tilt_from_state(s)
      stab = soft_stability_reward(
        tilt,
        limit_rad=cfg.stability_limit_rad,
        soft_rad=cfg.stability_soft_rad,
      )
      r = float(cfg.w_stability) * stab
      if cfg.clip_imag_reward:
        r = r.clamp(-clip, clip)
      if cfg.imag_soft_survival:
        surv = surv * (0.5 + 0.5 * (1.0 + stab)).clamp(0.05, 1.0)
      returns.append(discount * surv * r)
      discount = discount * cfg.gamma
      last_a = a

    feat_h = self.feat_fn(s, goals, geoms)
    v_h = self.critic(feat_h)
    j = torch.stack(returns, dim=0).sum(0) + discount * surv * v_h
    loss = -j.mean()

    if (
      teacher_body is not None
      and a0 is not None
      and float(cfg.w_teacher) > 0.0
      and teacher_body.shape[-1] == self.body_dim
    ):
      tgt = teacher_body[:, 0].detach()
      teacher_mse = F.mse_loss(a0, tgt)
      loss = loss + float(cfg.w_teacher) * teacher_mse

    out = self._step_or_skip(
      loss=loss, opt=self.opt_actor, module=self.actor, key="actor_loss"
    )
    mapped = {
      "actor_loss": out.get("actor_loss", float("nan")),
      "actor_grad_norm": float(out.get("actor_loss_grad_norm", 0.0) or 0.0),
      "J_mean": float(j.detach().mean()) if _finite(j) else float("nan"),
    }
    if teacher_mse is not None and _finite(teacher_mse):
      mapped["teacher_mse"] = float(teacher_mse.detach().item())
    if a0 is not None:
      mapped["actor_a_body_norm"] = float(a0.detach().norm(dim=-1).mean().item())

    for p in self.wm.parameters():
      p.requires_grad_(True)
    for p in self.critic.parameters():
      p.requires_grad_(True)
    return mapped

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
    bad = any(
      (k in logs and isinstance(logs[k], float) and not math_isfinite(logs[k]))
      for k in ("wm_loss", "critic_loss", "actor_loss", "J_mean")
    )
    if bad:
      self.nan_streak += 1
      logs["nan_streak"] = float(self.nan_streak)
      if self.cfg.nan_guard and self.nan_streak >= int(self.cfg.nan_reload_patience):
        self.restore_good()
        self.decay_lrs(self.cfg.nan_lr_decay)
        self.nan_streak = 0
        logs["nan_reloaded"] = 1.0
    else:
      self.nan_streak = 0
      self.snapshot_good()
      logs["nan_streak"] = 0.0
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
      "action_semantics": self.action_semantics,
      "phase": int(self.cfg.phase),
      "body_dim": self.body_dim,
      "experiment": "stage12_bptt",
    }

  def load_state_dict(self, sd: dict) -> None:
    if sd.get("experiment") not in (None, "stage12_bptt"):
      raise RuntimeError(f"Not a stage12_bptt ckpt: experiment={sd.get('experiment')!r}")
    sem = sd.get("action_semantics")
    if sem is not None and sem != self.action_semantics:
      raise RuntimeError(
        f"action_semantics mismatch ckpt={sem!r} vs {self.action_semantics!r}"
      )
    if int(sd.get("body_dim", self.body_dim)) != self.body_dim:
      raise RuntimeError("body_dim mismatch — rebuild trainer for this phase")
    self.actor.load_state_dict(sd["actor"])
    self.critic.load_state_dict(sd["critic"])
    self.critic_tgt.load_state_dict(sd["critic_tgt"])
    self.wm.load_state_dict(sd["wm"])
    self.opt_actor.load_state_dict(sd["opt_actor"])
    self.opt_critic.load_state_dict(sd["opt_critic"])
    self.opt_wm.load_state_dict(sd["opt_wm"])
    self.iter = int(sd.get("iter", 0))
    self.snapshot_good()
