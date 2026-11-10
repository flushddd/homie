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
from datetime import datetime
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
  p.add_argument(
    "--log-dir",
    type=str,
    default="logs/rsl_rl/homie_g1",
    help="Experiment root; each run creates <stamp>_<run-name>/ under this.",
  )
  p.add_argument(
    "--run-name",
    type=str,
    default="stage3_bptt",
    help="Suffix for the dated run folder (mjlab/rsl_rl style).",
  )
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
  p.add_argument(
    "--control-mode",
    type=str,
    default="hierarchical_subgoal",
    choices=("hierarchical_subgoal", "ee_residual"),
    help="hierarchical_subgoal: actor sets grasp subgoal for frozen loco+IK; "
    "ee_residual: legacy Δp on EE IK.",
  )
  p.add_argument("--save-every", type=int, default=200)
  return p.parse_args()


def _print_iter_log(
  *,
  it: int,
  tot_iter: int,
  collect_s: float,
  update_s: float,
  iter_s: float,
  env_steps: float,
  tot_timesteps: int,
  tot_time: float,
  logs: dict[str, float],
  width: int = 80,
  pad: int = 35,
) -> None:
  """Homie/rsl_rl-style console block (phase-3 PPO terms under Mean episode …)."""
  title = f" \033[1m Stage3 BPTT iteration {it}/{tot_iter - 1} \033[0m "
  fps = env_steps / max(iter_s, 1e-6)
  done = it + 1
  remaining = max(tot_iter - done, 0)
  eta = tot_time / max(done, 1) * remaining
  lines = [
    "#" * width,
    title.center(width, " "),
    "",
    (
      f"{'Computation:':>{pad}} {fps:.0f} steps/s "
      f"(collection: {collect_s:.3f}s, learning {update_s:.3f}s)"
    ),
  ]
  core = [
    ("collect_rew_mean", "Stage3 reward"),
    ("ppo_step_rew_mean", "PPO step reward"),
    ("collect_neg_clearance_frac", "Neg clearance frac"),
    ("collect_present_frac", "Present frac"),
    ("replay_size", "Replay size"),
    ("ppo_ep_log_count", "PPO ep log count"),
    ("wm_loss", "World-model loss"),
    ("critic_loss", "Critic loss"),
    ("actor_loss", "Actor loss"),
    ("actor_grad_norm", "Actor grad norm"),
    ("J_mean", "Imagination J mean"),
  ]
  for key, label in core:
    if key in logs:
      lines.append(f"{f'{label}:':>{pad}} {float(logs[key]):.4f}")
  # PPO Episode_Reward/* (and other extras) collected for monitoring.
  for key in sorted(k for k in logs if k.startswith("ppo/") and k != "ppo_ep_log_count"):
    ep_key = key[len("ppo/") :]
    lines.append(f"{f'Mean episode {ep_key}:':>{pad}} {float(logs[key]):.4f}")
  lines.extend(
    [
      "-" * width,
      f"{'Total timesteps:':>{pad}} {tot_timesteps}",
      f"{'Iteration time:':>{pad}} {iter_s:.2f}s",
      f"{'Total time:':>{pad}} {tot_time:.2f}s",
      f"{'ETA:':>{pad}} {eta:.1f}s",
      "",
    ]
  )
  print("\n".join(lines))


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

  # mjlab/rsl_rl layout: <log-dir>/<YYYY-mm-dd_HH-MM-SS>_<run-name>/
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_name = (args.run_name or "stage3_bptt").strip() or "stage3_bptt"
  log_dir = Path(args.log_dir) / f"{stamp}_{run_name}"
  log_dir.mkdir(parents=True, exist_ok=True)

  cfg = Stage3BpttCfg(
    device=args.device,
    stage2_checkpoint=args.stage2_checkpoint,
    log_dir=str(log_dir),
    curriculum=cur,
    collect_steps_per_iter=args.collect_per_iter,
    replay_capacity=max(200_000, need_cap),
    control_mode=str(args.control_mode),
  )
  print(
    f"[stage3_bptt] num_envs={n_envs} replay_capacity={cfg.replay_capacity} "
    f"(~{cfg.replay_capacity // n_envs} steps/env) control_mode={cfg.control_mode}"
  )
  print(f"[INFO] Logging experiment in directory: {log_dir}")

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
    collector.obs_dim, cfg.actor_hidden, collector.actor_action_scale
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

  def _save_ckpt(tag: int | str) -> Path:
    ckpt = log_dir / f"model_{tag}.pt"
    payload = trainer.state_dict()
    payload["stage2_checkpoint"] = args.stage2_checkpoint
    payload["log_dir"] = str(log_dir)
    payload["control_mode"] = cfg.control_mode
    torch.save(payload, ckpt)
    print(f"[info] saved {ckpt}")
    return ckpt

  # Warmup snapshot (iter 0), then every --save-every (launch: 200).
  _save_ckpt(0)

  tot_timesteps = int(args.warmup_collect) * n_envs
  tot_time = warm_s
  tot_iter = int(args.iters)
  save_every = max(1, int(args.save_every))
  for it in range(tot_iter):
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
    tot_timesteps += int(env_steps)
    tot_time += iter_s
    merged = {**clogs, **tlogs}
    if merged.get("nan_reloaded", 0.0) > 0.5:
      print(
        f"[warn] NaN streak → restored last-good weights; "
        f"wm_lr={merged.get('wm_lr', float('nan')):.2e}"
      )
      _save_ckpt(f"last_good_iter{it}")
    _print_iter_log(
      it=it,
      tot_iter=tot_iter,
      collect_s=collect_s,
      update_s=update_s,
      iter_s=iter_s,
      env_steps=env_steps,
      tot_timesteps=tot_timesteps,
      tot_time=tot_time,
      logs=merged,
    )
    if (it + 1) % save_every == 0:
      # Avoid overwriting a finite ckpt with NaN weights.
      if any(
        isinstance(merged.get(k), float) and merged[k] != merged[k]
        for k in ("wm_loss", "critic_loss", "actor_loss")
      ):
        print(f"[warn] skip periodic save at iter {it+1}: non-finite train losses")
      else:
        _save_ckpt(it + 1)

  if tot_iter % save_every != 0:
    _save_ckpt(tot_iter)

  print(f"[ok] real train loop finished (curriculum {cur}) log_dir={log_dir}")
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
