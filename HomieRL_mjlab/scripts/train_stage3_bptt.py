#!/usr/bin/env python3
"""Stage-3 BPTT trainer entry (does not replace PPO Phase-3).

Examples::

  .venv/bin/python scripts/train_stage3_bptt.py --check
  .venv/bin/python scripts/train_stage3_bptt.py --synthetic-smoke

  # Real sim loop (curriculum A, attractor EE, frozen phase-2 loco)
  .venv/bin/python scripts/train_stage3_bptt.py \\
    --stage2-checkpoint logs/rsl_rl/homie_g1/phase2/model_15400\\(2\\).pt \\
    --num-envs 16 --device cuda:0 --iters 100 --warmup-collect 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))


def _parse() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Homie Stage-3 BPTT trainer")
  p.add_argument("--check", action="store_true")
  p.add_argument("--synthetic-smoke", action="store_true")
  p.add_argument("--stage2-checkpoint", type=str, default="")
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--log-dir", type=str, default="logs/stage3_bptt")
  p.add_argument(
    "--curriculum",
    type=str,
    default="B",
    help="A=no container; B=crates for SDF/clearance BPTT",
  )
  p.add_argument("--iters", type=int, default=50)
  p.add_argument("--warmup-collect", type=int, default=100)
  p.add_argument("--collect-per-iter", type=int, default=50)
  p.add_argument("--ee-mode", type=str, default="attractor", choices=("attractor", "actor"))
  p.add_argument("--save-every", type=int, default=10)
  return p.parse_args()


def run_checks() -> int:
  from homierl_mjlab.rl.stage3_bptt.checks import run_all_checks

  out = run_all_checks()
  print(json.dumps(out, indent=2))
  g = out["bptt_graph"]
  assert g["actor_grad_sum"] > 0.0
  assert g["wm_has_grad"] == 0.0
  print("[ok] offline checks passed")
  return 0


def run_synthetic_smoke(device: str) -> int:
  from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
  from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer

  cfg = Stage3BpttCfg(
    device=device,
    wm_batch=32,
    wm_window=4,
    imagine_horizon=4,
    wm_updates_per_iter=1,
    critic_updates_per_iter=1,
    actor_updates_per_iter=1,
    replay_capacity=5000,
  )
  state_dim, control_dim = 20, 10
  goal_dim, geom_dim = 3, 4
  obs_dim = 12
  feat_dim = state_dim + goal_dim + geom_dim

  def control_fn(state, delta_p, goal, geom):
    u = torch.zeros(state.shape[0], control_dim, device=state.device, dtype=state.dtype)
    u = u.clone()
    u[:, :3] = delta_p
    return u

  def obs_fn(state, goal, geom, last_a):
    return torch.cat((state[..., :6], goal, last_a), dim=-1)

  def feat_fn(state, goal, geom):
    return torch.cat((state, goal, geom), dim=-1)

  def geom_fn(state, goal, geom):
    return state[..., :3], torch.ones(state.shape[0], 3, device=state.device) * 0.2

  trainer = Stage3BpttTrainer(
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
  )
  trainer.num_envs_hint = 1
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
      truncated=torch.tensor(
        [1.0 if t == 63 else 0.0], device=trainer.device
      ),
      goal=torch.zeros(1, goal_dim, device=trainer.device),
      geom=torch.zeros(1, geom_dim, device=trainer.device),
      env_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
      episode_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
    )
    s = ns
  print(json.dumps(trainer.train_iteration(), indent=2))
  print("[ok] synthetic smoke iteration done")
  return 0


def run_real(args: argparse.Namespace) -> int:
  from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
  from homierl_mjlab.rl.stage3_bptt.collector import Stage3SimCollector
  from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor
  from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer

  cur = args.curriculum.upper()
  if cur not in ("A", "B"):
    print(f"[warn] unknown curriculum {args.curriculum}; using B")
    cur = "B"
  print(f"[stage3_bptt] curriculum={cur} (A=no crate, B=crate+SDF clearance)")

  n_envs = int(args.num_envs)
  # SuperMimic: replay = num_envs × 1000 steps/env (not a fixed 200k total).
  # Also keep capacity a multiple of N so circular wrap never splits a step.
  steps_per_env = max(1000, int(args.warmup_collect) + 64)
  need_cap = n_envs * steps_per_env
  if n_envs >= 512:
    print(
      f"[warn] num_envs={n_envs} → replay ~{need_cap} transitions "
      f"(SuperMimic-style {steps_per_env} steps/env); needs substantial VRAM."
    )

  cfg = Stage3BpttCfg(
    device=args.device,
    stage2_checkpoint=args.stage2_checkpoint,
    log_dir=args.log_dir,
    curriculum=cur,
    collect_steps_per_iter=args.collect_per_iter,
    replay_capacity=max(200_000, need_cap),
  )
  log_dir = Path(args.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  print(
    f"[stage3_bptt] num_envs={n_envs} replay_capacity={cfg.replay_capacity} "
    f"(~{cfg.replay_capacity // n_envs} steps/env)"
  )

  actor = None
  # obs_dim set after collector builds (depends on sdf_query_count)
  collector = Stage3SimCollector(
    cfg,
    stage2_checkpoint=args.stage2_checkpoint,
    num_envs=n_envs,
    ee_mode=args.ee_mode if args.ee_mode == "attractor" else "attractor",
    actor=None,
  )
  actor = Stage3Actor(
    collector.obs_dim, cfg.actor_hidden, cfg.action_scale_m
  ).to(collector.device)
  if args.ee_mode == "actor":
    collector.ee_mode = "actor"
    collector.actor = actor

  control_fn, obs_fn, feat_fn, geom_fn = collector.make_fns()
  trainer = Stage3BpttTrainer(
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
  # Share the same actor weights the collector uses when ee_mode=actor.
  trainer.actor = actor
  trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=cfg.actor_lr)

  print(f"[info] warmup collect {args.warmup_collect} steps …")
  t0 = time.perf_counter()
  logs = collector.collect(trainer.replay, args.warmup_collect)
  warm_s = time.perf_counter() - t0
  print(json.dumps({**logs, "wall_s": warm_s}, indent=2))
  print(f"[info] warmup wall {warm_s:.1f}s")

  for it in range(int(args.iters)):
    t_iter = time.perf_counter()
    t_c = time.perf_counter()
    clogs = collector.collect(trainer.replay, cfg.collect_steps_per_iter)
    collect_s = time.perf_counter() - t_c
    t_u = time.perf_counter()
    tlogs = trainer.train_iteration()
    update_s = time.perf_counter() - t_u
    iter_s = time.perf_counter() - t_iter
    n_env = int(collector.env.num_envs)
    env_steps = float(cfg.collect_steps_per_iter * n_env)
    merged = {
      **clogs,
      **tlogs,
      "wall_s": iter_s,
      "collect_s": collect_s,
      "update_s": update_s,
      "env_steps": env_steps,
      "env_steps_per_s": env_steps / max(iter_s, 1e-6),
    }
    print(f"[iter {it}] " + json.dumps(merged))
    if (it + 1) % max(1, args.save_every) == 0:
      ckpt = log_dir / f"stage3_bptt_{it+1}.pt"
      payload = trainer.state_dict()
      payload["stage2_checkpoint"] = args.stage2_checkpoint
      torch.save(payload, ckpt)
      print(f"[info] saved {ckpt}")

  print(f"[ok] real train loop finished (curriculum {cur})")
  return 0


def main() -> int:
  args = _parse()
  if args.check:
    return run_checks()
  if args.synthetic_smoke:
    return run_synthetic_smoke(args.device)
  if not args.stage2_checkpoint:
    print("error: --stage2-checkpoint required for real train", file=sys.stderr)
    return 2
  return run_real(args)


if __name__ == "__main__":
  raise SystemExit(main())
