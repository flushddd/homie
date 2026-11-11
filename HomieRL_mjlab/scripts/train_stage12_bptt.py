#!/usr/bin/env python3
"""Stage-1/2 SuperMimic BPTT trainer (experimental; parallel to HIM-PPO).

Examples::

  .venv/bin/python scripts/train_stage12_bptt.py --check
  .venv/bin/python scripts/train_stage12_bptt.py --synthetic-smoke --device cpu
  .venv/bin/python scripts/train_stage12_bptt.py --phase 1 --num-envs 16 --iters 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))


def _parse() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Homie Stage-1/2 SuperMimic BPTT (experiment)")
  p.add_argument("--check", action="store_true")
  p.add_argument("--synthetic-smoke", action="store_true")
  p.add_argument("--phase", type=int, default=1, choices=(1, 2))
  p.add_argument(
    "--phase1-checkpoint",
    type=str,
    default="",
    help="Optional frozen phase-1 ckpt for phase-2 teacher BC",
  )
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--log-dir", type=str, default="logs/rsl_rl/homie_g1")
  p.add_argument("--run-name", type=str, default="")
  p.add_argument("--iters", type=int, default=1000)
  p.add_argument("--warmup-collect", type=int, default=100)
  p.add_argument("--collect-per-iter", type=int, default=50)
  p.add_argument("--save-every", type=int, default=200)
  p.add_argument("--w-loco", type=float, default=None)
  p.add_argument("--w-teacher", type=float, default=None)
  return p.parse_args()


def run_checks() -> int:
  from homierl_mjlab.rl.stage12_bptt.checks import run_all_checks

  out = run_all_checks()
  print(json.dumps(out, indent=2))
  print("[ok] stage12 offline checks passed")
  return 0


def run_synthetic_smoke(device: str) -> int:
  from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
  from homierl_mjlab.rl.stage12_bptt.trainer import Stage12BpttTrainer

  cfg = Stage12BpttCfg(
    device=device,
    phase=1,
    wm_batch=32,
    wm_window=4,
    imagine_horizon=4,
    wm_updates_per_iter=1,
    critic_updates_per_iter=1,
    actor_updates_per_iter=1,
    replay_capacity=5000,
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
    return torch.cat((state[..., -13:], last_a[..., : cfg.body_dim]), dim=-1)

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
  print(json.dumps(trainer.train_iteration(), indent=2))
  print("[ok] stage12 synthetic smoke done")
  return 0


def run_real(args: argparse.Namespace) -> int:
  from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
  from homierl_mjlab.rl.stage12_bptt.collector import Stage12SimCollector
  from homierl_mjlab.rl.stage12_bptt.trainer import Stage12BpttTrainer

  n_envs = int(args.num_envs)
  steps_per_env = max(1000, int(args.warmup_collect) + 64)
  need_cap = n_envs * steps_per_env
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_name = (args.run_name or f"stage12_bptt_p{args.phase}").strip()
  log_dir = Path(args.log_dir) / f"{stamp}_{run_name}"
  log_dir.mkdir(parents=True, exist_ok=True)

  cfg = Stage12BpttCfg(
    device=args.device,
    phase=int(args.phase),
    phase1_checkpoint=str(args.phase1_checkpoint or ""),
    log_dir=str(log_dir),
    collect_steps_per_iter=int(args.collect_per_iter),
    replay_capacity=max(200_000, need_cap),
  )
  cfg.resolve_dims()
  if args.w_loco is not None:
    cfg.w_loco = float(args.w_loco)
  if args.w_teacher is not None:
    cfg.w_teacher = float(args.w_teacher)

  print(
    f"[stage12_bptt] EXPERIMENT parallel to HIM-PPO / Stage-3 | "
    f"phase={cfg.phase} task={cfg.task_id()} body_dim={cfg.body_dim} "
    f"semantics={cfg.action_semantics}"
  )
  print(
    f"[stage12_bptt] num_envs={n_envs} replay={cfg.replay_capacity} "
    f"w_loco={cfg.w_loco} w_teacher={cfg.w_teacher}"
  )
  print(f"[INFO] Logging experiment in directory: {log_dir}")
  writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)

  collector = Stage12SimCollector(cfg, num_envs=n_envs, actor=None)
  control_fn, obs_fn, feat_fn, geom_fn = collector.make_fns()
  trainer = Stage12BpttTrainer(
    cfg,
    state_dim=collector.state_dim,
    control_dim=collector.control_dim,
    obs_dim=collector.obs_dim,
    feat_dim=collector.feat_dim,
    goal_dim=collector.goal_dim,
    geom_dim=collector.geom_dim,
    control_fn=control_fn,
    obs_fn=obs_fn,
    feat_fn=feat_fn,
    geom_fn=geom_fn,
    num_envs=int(collector.env.num_envs),
  )
  collector.actor = trainer.actor
  print(
    f"[stage12_bptt] obs_dim={collector.obs_dim} state_dim={collector.state_dim} "
    f"env_action_dim={collector.env_action_dim}"
  )

  print(f"[info] warmup collect {args.warmup_collect} steps …")
  t0 = time.perf_counter()
  logs = collector.collect(trainer.replay, args.warmup_collect)
  print(json.dumps({**logs, "wall_s": time.perf_counter() - t0}, indent=2))

  def _save(tag: int | str) -> Path:
    ckpt = log_dir / f"model_{tag}.pt"
    payload = trainer.state_dict()
    payload["log_dir"] = str(log_dir)
    torch.save(payload, ckpt)
    print(f"[info] saved {ckpt}")
    return ckpt

  _save(0)
  tot_iter = int(args.iters)
  save_every = max(1, int(args.save_every))
  for it in range(tot_iter):
    clogs = collector.collect(trainer.replay, cfg.collect_steps_per_iter)
    tlogs = trainer.train_iteration()
    merged = {**clogs, **tlogs}
    print(
      f"[stage12_bptt] iter={it}/{tot_iter - 1} "
      f"rew={merged.get('collect_rew_mean', float('nan')):.4f} "
      f"ppo={merged.get('ppo_step_rew_mean', float('nan')):.4f} "
      f"ep_len={merged.get('mean_episode_length', 0.0):.1f} "
      f"fall={merged.get('fall_frac', 0.0):.3f} "
      f"wm={merged.get('wm_loss', float('nan')):.4f} "
      f"actor={merged.get('actor_loss', float('nan')):.4f}"
    )
    for k, v in merged.items():
      if isinstance(v, (int, float)) and v == v:
        writer.add_scalar(f"Train/{k}", float(v), it)
    writer.flush()
    if (it + 1) % save_every == 0:
      _save(it + 1)
  if tot_iter % save_every != 0:
    _save(tot_iter)
  writer.close()
  print(f"[ok] stage12_bptt finished phase={cfg.phase} log_dir={log_dir}")
  return 0


def main() -> int:
  args = _parse()
  if args.check:
    return run_checks()
  if args.synthetic_smoke:
    return run_synthetic_smoke(args.device)
  return run_real(args)


if __name__ == "__main__":
  raise SystemExit(main())
