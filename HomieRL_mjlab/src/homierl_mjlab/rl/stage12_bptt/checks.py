"""Offline checks for Stage-1/2 SuperMimic scaffold."""

from __future__ import annotations

import torch

from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
from homierl_mjlab.rl.stage12_bptt.networks import LocoActor, LocoCritic, WorldModel
from homierl_mjlab.rl.stage12_bptt.trainer import Stage12BpttTrainer


def check_loco_actor_grads(body_dim: int = 12) -> dict[str, float]:
  obs_dim = 32
  actor = LocoActor(obs_dim, body_dim=body_dim, hidden=(64, 64))
  obs = torch.randn(4, obs_dim)
  a = actor(obs)
  loss = a.pow(2).mean()
  loss.backward()
  last = actor.net[-1]
  assert isinstance(last, torch.nn.Linear) and last.weight.grad is not None
  return {
    "body_dim": float(body_dim),
    "grad_sum": float(last.weight.grad.abs().sum().item()),
    "a_norm": float(a.detach().norm(dim=-1).mean().item()),
  }


def check_bptt_graph() -> dict[str, float]:
  cfg = Stage12BpttCfg(
    device="cpu",
    phase=1,
    wm_batch=16,
    wm_window=4,
    imagine_horizon=4,
    wm_updates_per_iter=1,
    critic_updates_per_iter=1,
    actor_updates_per_iter=1,
    replay_capacity=2000,
    w_teacher=0.0,
  )
  cfg.resolve_dims()
  state_dim, control_dim = 40, 20
  goal_dim, geom_dim = 3, 1
  obs_dim = 13 + cfg.body_dim
  feat_dim = state_dim + goal_dim + geom_dim

  def control_fn(state, action, goal, geom):
    u = state[..., :control_dim].clone()
    u[:, : cfg.body_dim] = action[..., : cfg.body_dim]
    return u

  def obs_fn(state, goal, geom, last_a):
    root = state[..., -13:]
    return torch.cat((root, last_a[..., : cfg.body_dim]), dim=-1)

  def feat_fn(state, goal, geom):
    return torch.cat((state, goal, geom), dim=-1)

  def geom_fn(state, goal, geom):
    return state[..., -3:], torch.ones(state.shape[0], 3, device=state.device)

  trainer = Stage12BpttTrainer(
    cfg,
    state_dim=state_dim,
    control_dim=control_dim,
    obs_dim=obs_dim,
    feat_dim=feat_dim,
    goal_dim=goal_dim,
    geom_dim=geom_dim,
    control_fn=control_fn,
    obs_fn=obs_fn,
    feat_fn=feat_fn,
    geom_fn=geom_fn,
    num_envs=1,
  )
  s = torch.randn(1, state_dim, device=trainer.device)
  for t in range(64):
    u = torch.randn(1, control_dim, device=trainer.device) * 0.01
    ns = s + 0.01 * torch.randn_like(s)
    trainer.replay.add_batch(
      state=s,
      control=u,
      next_state=ns,
      reward=torch.zeros(1, device=trainer.device),
      terminated=torch.zeros(1, device=trainer.device),
      truncated=torch.tensor([1.0 if t == 63 else 0.0], device=trainer.device),
      goal=torch.zeros(1, goal_dim, device=trainer.device),
      geom=torch.zeros(1, geom_dim, device=trainer.device),
      present=torch.zeros(1, device=trainer.device),
      env_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
      episode_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
      actor_action=torch.zeros(1, cfg.body_dim, device=trainer.device),
    )
    s = ns
  logs = trainer.train_iteration()
  return {
    "wm_loss": float(logs.get("wm_loss", float("nan"))),
    "critic_loss": float(logs.get("critic_loss", float("nan"))),
    "actor_loss": float(logs.get("actor_loss", float("nan"))),
    "actor_grad_norm": float(logs.get("actor_grad_norm", 0.0)),
    "J_mean": float(logs.get("J_mean", float("nan"))),
  }


def run_all_checks() -> dict[str, object]:
  out: dict[str, object] = {}
  g12 = check_loco_actor_grads(12)
  g14 = check_loco_actor_grads(14)
  out["actor_grads_p1"] = g12
  out["actor_grads_p2"] = g14
  assert g12["grad_sum"] > 0.0 and g14["grad_sum"] > 0.0
  graph = check_bptt_graph()
  out["bptt_graph"] = graph
  assert graph["actor_grad_norm"] > 0.0 or abs(graph["actor_loss"]) > 0.0
  # Sanity: WorldModel / Critic importable
  _ = WorldModel(8, 4, ctx_dim=2, hidden=(16,))
  _ = LocoCritic(10, (16,))
  out["imports_ok"] = 1.0
  return out


if __name__ == "__main__":
  import json

  print(json.dumps(run_all_checks(), indent=2))
