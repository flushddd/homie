"""World-model fit + critic TD + short-horizon BPTT actor updates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor, Stage3Critic, WorldModel
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import (
  soft_stability_reward,
  stage3_reward,
  tilt_angle_from_root_quat,
)


def _finite(x: torch.Tensor) -> bool:
  return bool(torch.isfinite(x).all().item())


def _sanitize(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
  return tuple(
    torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0) for t in tensors
  )


def _params_finite(module: nn.Module) -> bool:
  for p in module.parameters():
    if not torch.isfinite(p).all():
      return False
  return True


def math_isfinite(x: float) -> bool:
  return x == x and abs(x) != float("inf")


class Stage3BpttTrainer:
  """Alternating: WM → critic → BPTT actor (SuperMimic-style short AC).

  Imagination control synthesis (v1):
    - ``control = stitch(logged_loco_u, dls(arm | delta_p))`` is provided by
      ``control_fn(state, delta_p, goal, geom) -> control``.
    - Palm / clearance for rewards via ``geom_fn(state, goal, geom)``.
    - ``present`` from replay gates safe/margin like real collect (P0).
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
    palm_step_fn: Callable[..., torch.Tensor] | None = None,
    num_envs: int = 1,
  ):
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.control_fn = control_fn
    self.obs_fn = obs_fn
    self.feat_fn = feat_fn
    self.geom_fn = geom_fn
    self.palm_step_fn = palm_step_fn
    self.num_envs_hint = int(num_envs)

    scale = (
      float(cfg.subgoal_offset_scale_m)
      if str(cfg.control_mode) == "hierarchical_subgoal"
      else float(cfg.action_scale_m)
    )
    self.actor = Stage3Actor(obs_dim, cfg.actor_hidden, scale).to(self.device)
    self.critic = Stage3Critic(feat_dim, cfg.critic_hidden).to(self.device)
    self.critic_tgt = Stage3Critic(feat_dim, cfg.critic_hidden).to(self.device)
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
    )
    self.last_action = torch.zeros(1, 3, device=self.device)
    self.iter = 0
    self.nan_streak = 0
    self._last_good: dict | None = None
    self.snapshot_good()

  def snapshot_good(self) -> None:
    """CPU copy of weights used to recover after NaN updates."""
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
    """Backward + step with NaN guard; restore module if params blow up."""
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

  def _tilt_from_state(self, state: torch.Tensor) -> torch.Tensor:
    """Root tilt from packed state ``…|pos3|quat_wxyz4|lin3|ang3|palm3``."""
    quat = state[..., -16:-12]
    return tilt_angle_from_root_quat(quat)

  def update_world_model(self) -> dict[str, float]:
    cfg = self.cfg
    states, controls, _, _, _, goals, geoms, _ = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=getattr(self, "num_envs_hint", None)
    )
    states, controls, goals, geoms = _sanitize(states, controls, goals, geoms)
    ctx = torch.cat((goals, geoms), dim=-1)
    pred = []
    s = states[:, 0]
    for t in range(cfg.wm_window):
      s = self.wm(s, controls[:, t], ctx)
      pred.append(s)
    pred_s = torch.stack(pred, dim=1)
    target = states[:, 1:]
    loss = F.huber_loss(pred_s, target, delta=float(cfg.wm_huber_delta))
    return self._step_or_skip(loss=loss, opt=self.opt_wm, module=self.wm, key="wm_loss")

  def update_critic(self) -> dict[str, float]:
    cfg = self.cfg
    (
      states,
      controls,
      rewards,
      terms,
      _truncs,
      goals,
      geoms,
      _present,
    ) = self.replay.sample_contiguous_windows(
      cfg.wm_batch, cfg.wm_window, num_envs_hint=getattr(self, "num_envs_hint", None)
    )
    states, controls, rewards, terms, goals, geoms = _sanitize(
      states, controls, rewards, terms, goals, geoms
    )
    clip = float(cfg.reward_clip)
    if cfg.clip_critic_reward:
      clipped = rewards.clamp(-clip, clip)
      clip_frac = float((rewards.abs() > clip).float().mean().item())
    else:
      clipped = rewards
      clip_frac = 0.0
    b, h, _ = controls.shape
    feats = []
    for t in range(h + 1):
      feats.append(self.feat_fn(states[:, t], goals, geoms))
    values = torch.stack([self.critic(f) for f in feats], dim=1)
    with torch.no_grad():
      v_tgt = torch.stack([self.critic_tgt(f) for f in feats], dim=1)
      returns = torch.zeros(b, h, device=self.device)
      gae = torch.zeros(b, device=self.device)
      for t in reversed(range(h)):
        # Bootstrap through time_out (truncated); stop only on true terminated.
        mask = 1.0 - terms[:, t]
        delta = clipped[:, t] + cfg.gamma * mask * v_tgt[:, t + 1] - v_tgt[:, t]
        gae = delta + cfg.gamma * cfg.lam * mask * gae
        returns[:, t] = gae + v_tgt[:, t]
    loss = F.mse_loss(values[:, :-1], returns.detach())
    out = self._step_or_skip(
      loss=loss, opt=self.opt_critic, module=self.critic, key="critic_loss"
    )
    out["critic_reward_clip_frac"] = clip_frac
    if out.get("critic_loss_skipped") or out.get("critic_loss_restored"):
      return out
    with torch.no_grad():
      for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
        pt.data.mul_(1.0 - cfg.target_critic_tau).add_(cfg.target_critic_tau * p.data)
    return out

  def update_actor_bptt(self) -> dict[str, float]:
    """Freeze WM + critic **weights**; grads still flow through WM/critic inputs."""
    cfg = self.cfg
    states, _, _, _, _, goals, geoms, present = self.replay.sample_contiguous_windows(
      max(32, cfg.wm_batch // 4),
      1,
      num_envs_hint=getattr(self, "num_envs_hint", None),
    )
    states, goals, geoms, present = _sanitize(states, goals, geoms, present)
    s = states[:, 0].detach().requires_grad_(True)
    ctx = torch.cat((goals, geoms), dim=-1)
    last_a = torch.zeros(s.shape[0], 3, device=self.device)
    returns = []
    discount = torch.ones(s.shape[0], device=self.device)
    surv = torch.ones(s.shape[0], device=self.device)
    present_b = present.reshape(s.shape[0]) if cfg.imag_use_present else None
    term_acc: dict[str, list[torch.Tensor]] = {}
    clip = float(cfg.reward_clip)
    clip_hits = 0
    clip_total = 0

    for p in self.wm.parameters():
      p.requires_grad_(False)
    for p in self.critic.parameters():
      p.requires_grad_(False)

    use_palm = bool(cfg.use_palm_step) and self.palm_step_fn is not None
    for _t in range(cfg.imagine_horizon):
      obs = self.obs_fn(s, goals, geoms, last_a)
      obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
      a = self.actor(obs)
      u = self.control_fn(s, a, goals, geoms)
      s_wm = self.wm(s, u, ctx)
      if use_palm:
        palm_kin = self.palm_step_fn(s, a, goals, geoms)
        s = torch.cat((s_wm[..., :-3], palm_kin), dim=-1)
      else:
        s = s_wm
      palm, clearance = self.geom_fn(s, goals, geoms)
      stab = None
      if float(cfg.w_stability) != 0.0:
        tilt = self._tilt_from_state(s)
        stab = soft_stability_reward(
          tilt,
          limit_rad=cfg.stability_limit_rad,
          soft_rad=cfg.stability_soft_rad,
        )
      r, terms = stage3_reward(
        palm_pos=palm,
        goal_pos=goals[..., :3],
        clearance=clearance,
        action=a,
        last_action=last_a,
        sigma_p=cfg.sigma_p,
        sigma_far=cfg.sigma_far,
        d_safe=cfg.d_safe,
        d_max=cfg.d_max,
        reach_threshold=cfg.reach_threshold,
        w_target=cfg.w_target,
        w_approach=cfg.w_approach,
        w_safe=cfg.w_safe,
        w_margin=cfg.w_margin,
        w_smooth=cfg.w_smooth,
        w_soft_success=cfg.w_soft_success,
        w_stability=cfg.w_stability,
        stability=stab,
        present=present_b,
        return_terms=True,
      )
      for k, v in terms.items():
        term_acc.setdefault(k, []).append(v.detach())
      if cfg.clip_imag_reward:
        clip_hits += int((r.abs() > clip).sum().item())
        clip_total += int(r.numel())
        r = r.clamp(-clip, clip)
      if cfg.imag_soft_survival:
        tilt = self._tilt_from_state(s)
        soft = soft_stability_reward(
          tilt,
          limit_rad=cfg.stability_limit_rad,
          soft_rad=cfg.stability_soft_rad,
        )
        surv = surv * (0.5 + 0.5 * (1.0 + soft)).clamp(0.05, 1.0)
      returns.append(discount * surv * r)
      discount = discount * cfg.gamma
      last_a = a

    feat_h = self.feat_fn(s, goals, geoms)
    v_h = self.critic(feat_h)
    j = torch.stack(returns, dim=0).sum(0) + discount * surv * v_h
    loss = -j.mean()

    out = self._step_or_skip(
      loss=loss, opt=self.opt_actor, module=self.actor, key="actor_loss"
    )
    gn = out.get("actor_loss_grad_norm", out.get("actor_grad_norm", 0.0))
    mapped = {
      "actor_loss": out.get("actor_loss", float("nan")),
      "actor_grad_norm": float(gn) if gn == gn else 0.0,
      "J_mean": float(j.detach().mean()) if _finite(j) else float("nan"),
      "imag_reward_clip_frac": float(clip_hits) / max(clip_total, 1),
      "imag_present_frac": float(present.mean().item()),
      "imag_use_palm_step": 1.0 if use_palm else 0.0,
    }
    for k, vs in term_acc.items():
      stacked = torch.stack(vs, dim=0)
      mapped[f"imag/{k}"] = float(stacked.float().mean().cpu())
    if out.get("actor_loss_skipped"):
      mapped["actor_skipped"] = 1.0
    if out.get("actor_loss_restored"):
      mapped["actor_restored"] = 1.0

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
    ) or any(
      logs.get(k, 0.0) > 0.5
      for k in ("wm_loss_restored", "critic_loss_restored", "actor_restored")
    )
    if bad:
      self.nan_streak += 1
      logs["nan_streak"] = float(self.nan_streak)
      if self.cfg.nan_guard and self.nan_streak >= int(self.cfg.nan_reload_patience):
        self.restore_good()
        self.decay_lrs(self.cfg.nan_lr_decay)
        self.nan_streak = 0
        logs["nan_reloaded"] = 1.0
        logs["wm_lr"] = float(self.opt_wm.param_groups[0]["lr"])
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
      "reward_schema": "stage3_v2_present_approach_stability",
    }

  def load_state_dict(self, sd: dict) -> None:
    schema = sd.get("reward_schema")
    if schema not in (None, "stage3_v2_present_approach_stability"):
      raise RuntimeError(
        f"Incompatible Stage3 checkpoint reward_schema={schema!r}; "
        "expected stage3_v2_present_approach_stability (or legacy None). "
        "Do not silent-load — restart from phase-2 loco ckpt."
      )
    self.actor.load_state_dict(sd["actor"])
    self.critic.load_state_dict(sd["critic"])
    self.critic_tgt.load_state_dict(sd["critic_tgt"])
    self.wm.load_state_dict(sd["wm"])
    self.opt_actor.load_state_dict(sd["opt_actor"])
    self.opt_critic.load_state_dict(sd["opt_critic"])
    self.opt_wm.load_state_dict(sd["opt_wm"])
    self.iter = int(sd.get("iter", 0))
    self.snapshot_good()
