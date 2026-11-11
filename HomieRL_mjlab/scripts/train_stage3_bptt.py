#!/usr/bin/env python3
"""Stage-3 BPTT trainer entry (does not replace PPO Phase-3).

Examples::

  .venv/bin/python scripts/train_stage3_bptt.py --check
  .venv/bin/python scripts/train_stage3_bptt.py --synthetic-smoke

  # Real sim loop (curriculum B, actor EE, frozen phase-2 loco)
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
from torch.utils.tensorboard import SummaryWriter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))


def _tb_log(writer: SummaryWriter, it: int, logs: dict[str, float], *, collect_s: float, update_s: float, env_steps: float, iter_s: float) -> None:
  """Write Stage-3 BPTT scalars (same layout as the short Nov-9 smoke events)."""
  fps = env_steps / max(iter_s, 1e-6)
  writer.add_scalar("Perf/total_fps", fps, it)
  writer.add_scalar("Perf/collection_time", collect_s, it)
  writer.add_scalar("Perf/learning_time", update_s, it)
  mapping = (
    ("collect_rew_mean", "Train/mean_reward"),
    ("ppo_step_rew_mean", "Train/ppo_step_reward"),
    ("mean_episode_length", "Train/mean_episode_length"),
    ("episode_length_count", "Train/episode_length_count"),
    ("collect_neg_clearance_frac", "Train/neg_clearance_frac"),
    ("collect_present_frac", "Train/present_frac"),
    ("reach_success_frac", "Train/reach_success_frac"),
    ("clear_reach_success_frac", "Train/clear_reach_success_frac"),
    ("mean_goal_distance", "Train/mean_goal_distance"),
    ("replay_size", "Train/replay_size"),
    ("wm_loss", "Loss/world_model"),
    ("critic_loss", "Loss/critic"),
    ("actor_loss", "Loss/actor"),
    ("actor_grad_norm", "Loss/actor_grad_norm"),
    ("J_mean", "Train/imagination_J"),
    ("imag_reward_clip_frac", "Train/imag_reward_clip_frac"),
    ("critic_reward_clip_frac", "Train/critic_reward_clip_frac"),
    ("nan_streak", "Train/nan_streak"),
    ("teacher_mse", "Loss/teacher_mse"),
    ("teacher_mse_collect", "Train/teacher_mse_collect"),
    ("actor_body_norm", "Train/actor_body_norm"),
    ("actor_ee_norm", "Train/actor_ee_norm"),
    ("w_loco", "Train/w_loco"),
    ("body_exec_teacher", "Train/body_exec_teacher"),
    ("body_exec_mix", "Train/body_exec_mix"),
    ("body_trust_region", "Train/body_trust_region"),
    ("real/ppo_loco", "RewardReal/ppo_loco"),
    ("real/w_loco", "RewardReal/w_loco"),
    ("actor_a_body_norm", "Train/actor_a_body_norm"),
    ("actor_a_ee_norm", "Train/actor_a_ee_norm"),
    ("curr_phase", "Curriculum/phase"),
    ("curr_arm_unlocked", "Curriculum/arm_unlocked"),
    ("curr_presence_prob", "Curriculum/presence_prob"),
    ("curr_reach_ema", "Curriculum/reach_ema"),
    ("curr_clear_reach_ema", "Curriculum/clear_reach_ema"),
    ("curr_teacher_mse_ema", "Curriculum/teacher_mse_ema"),
    ("curr_ep_len_ema", "Curriculum/ep_len_ema"),
  )
  for src, dst in mapping:
    if src in logs and isinstance(logs[src], (int, float)):
      v = float(logs[src])
      if v == v:  # skip NaN
        writer.add_scalar(dst, v, it)
  for key, val in logs.items():
    if not isinstance(val, (int, float)):
      continue
    v = float(val)
    if v != v:
      continue
    if key.startswith("ppo/"):
      writer.add_scalar(f"Episode/{key[len('ppo/'):]}", v, it)
    elif key.startswith("real/"):
      writer.add_scalar(f"RewardReal/{key[len('real/'):]}", v, it)
    elif key.startswith("imag/"):
      writer.add_scalar(f"RewardImag/{key[len('imag/'):]}", v, it)
  writer.flush()


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
    default="A",
    help="Static A=no crate / B=crates when --no-curriculum-auto. "
    "With auto (default), starts A and ramps boxes by reach metrics.",
  )
  p.add_argument(
    "--curriculum-auto",
    dest="curriculum_auto",
    action="store_true",
    default=True,
    help="Progressive body→arm→box (default on for body_ee).",
  )
  p.add_argument(
    "--no-curriculum-auto",
    dest="curriculum_auto",
    action="store_false",
    help="Disable progressive schedule; use --curriculum A|B only.",
  )
  p.add_argument(
    "--start-phase",
    type=str,
    default="body",
    choices=("body", "arm", "box"),
    help="Initial progressive phase when curriculum-auto is on.",
  )
  p.add_argument("--iters", type=int, default=100000)
  p.add_argument("--warmup-collect", type=int, default=100)
  p.add_argument("--collect-per-iter", type=int, default=50)
  p.add_argument(
    "--ee-mode",
    type=str,
    default="actor",
    choices=("attractor", "actor"),
    help="actor: collect with BPTT policy; attractor: heuristic data only",
  )
  p.add_argument(
    "--control-mode",
    type=str,
    default="body_ee_palm_delta",
    choices=("body_ee_palm_delta", "hierarchical_subgoal", "ee_residual"),
    help="body_ee_palm_delta (default): Actor [a_body(14)|a_ee(3)], "
    "p_sub=palm+delta, frozen phase-2 teacher on body; "
    "hierarchical_subgoal / ee_residual: legacy ablation only.",
  )
  p.add_argument(
    "--w-teacher",
    type=float,
    default=None,
    help="MSE weight for frozen phase-2 teacher on a_body (default cfg 2.0).",
  )
  p.add_argument(
    "--w-loco",
    type=float,
    default=None,
    help="Blend weight for Phase-1/2 env PPO step reward into Stage3 targets "
    "(default cfg 1.0).",
  )
  p.add_argument(
    "--ee-delta-scale-m",
    type=float,
    default=None,
    help="Palm-delta tanh scale in meters (default cfg 0.03).",
  )
  p.add_argument(
    "--body-trust-region",
    type=float,
    default=None,
    help="Max |exec_body - teacher| per dim (default cfg 0.20). "
    "Keeps loco from taking off when a_body is wrong; 0 disables.",
  )
  p.add_argument(
    "--body-exec-mix",
    type=float,
    default=None,
    help="While arm locked + body_exec_teacher: actor mix into sim "
    "(default cfg 0.0 = pure teacher).",
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
    ("mean_episode_length", "Mean episode length"),
    ("episode_length_count", "Episode length count"),
    ("collect_neg_clearance_frac", "Neg clearance frac"),
    ("collect_present_frac", "Present frac"),
    ("reach_success_frac", "Reach success frac"),
    ("clear_reach_success_frac", "Clear-reach success frac"),
    ("mean_goal_distance", "Mean goal distance"),
    ("replay_size", "Replay size"),
    ("ppo_ep_log_count", "PPO ep log count"),
    ("wm_loss", "World-model loss"),
    ("critic_loss", "Critic loss"),
    ("actor_loss", "Actor loss"),
    ("actor_grad_norm", "Actor grad norm"),
    ("J_mean", "Imagination J mean"),
    ("teacher_mse", "Teacher MSE"),
    ("teacher_mse_collect", "Teacher MSE (collect)"),
    ("actor_body_norm", "Actor body norm"),
    ("actor_ee_norm", "Actor EE norm"),
    ("body_exec_teacher", "Body exec teacher"),
    ("body_exec_mix", "Body exec mix"),
    ("body_trust_region", "Body trust region"),
    ("curr_phase", "Curriculum phase"),
    ("curr_arm_unlocked", "Arm unlocked"),
    ("curr_presence_prob", "Presence prob"),
    ("curr_reach_ema", "Reach EMA"),
    ("curr_clear_reach_ema", "Clear-reach EMA"),
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
    control_mode="body_ee_palm_delta",
    wm_batch=32,
    wm_window=4,
    imagine_horizon=4,
    wm_updates_per_iter=1,
    critic_updates_per_iter=1,
    actor_updates_per_iter=1,
    replay_capacity=5000,
    w_teacher=2.0,
  )
  state_dim, control_dim = 20, 10
  goal_dim, geom_dim = 3, 4
  body_dim, ee_dim = 14, 3
  obs_dim = 3 + 3 + 3 + 3 + body_dim + ee_dim + 4 + 6
  feat_dim = state_dim + goal_dim + geom_dim

  def control_fn(state, action, goal, geom):
    u = torch.zeros(state.shape[0], control_dim, device=state.device, dtype=state.dtype)
    u = u.clone()
    u[:, :3] = action[..., -3:]
    return u

  def obs_fn(state, goal, geom, last_a):
    b = state.shape[0]
    palm = state[..., :3]
    zero = torch.zeros_like(palm)
    la = last_a
    if la.shape[-1] != body_dim + ee_dim:
      pad = torch.zeros(b, body_dim + ee_dim, device=state.device, dtype=state.dtype)
      pad[..., : min(la.shape[-1], pad.shape[-1])] = la[..., : pad.shape[-1]]
      la = pad
    sdf = torch.zeros(b, 4, device=state.device, dtype=state.dtype)
    clr = torch.ones(b, 6, device=state.device, dtype=state.dtype) * 0.2
    return torch.cat(
      (palm, goal, goal - palm, zero, la, sdf, clr), dim=-1
    )

  def feat_fn(state, goal, geom):
    return torch.cat((state, goal, geom), dim=-1)

  def geom_fn(state, goal, geom):
    return state[..., :3], torch.ones(state.shape[0], 3, device=state.device) * 0.2

  def palm_step_fn(state, action, goal, geom):
    del goal, geom
    return state[..., -3:] + torch.clamp(action[..., -3:], -0.04, 0.04)

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
    palm_step_fn=palm_step_fn,
  )
  trainer.num_envs_hint = 1
  s = torch.randn(1, state_dim, device=trainer.device)
  for t in range(64):
    u = torch.randn(1, control_dim, device=trainer.device) * 0.01
    ns = s + 0.01 * torch.randn_like(s)
    a_act = torch.randn(1, body_dim + ee_dim, device=trainer.device) * 0.01
    teacher = torch.randn(1, body_dim, device=trainer.device) * 0.1
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
      present=torch.zeros(1, device=trainer.device),
      env_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
      episode_id=torch.zeros(1, device=trainer.device, dtype=torch.long),
      actor_action=a_act,
      teacher_body=teacher,
    )
    s = ns
  print(json.dumps(trainer.train_iteration(), indent=2))
  print("[ok] synthetic smoke iteration done")
  return 0


def _apply_start_phase(prog, start: str) -> None:
  from homierl_mjlab.rl.stage3_bptt.curriculum_progress import (
    PHASE_ARM,
    PHASE_BODY,
    PHASE_BOX,
  )

  name = str(start).lower()
  if name == "arm":
    prog.phase = PHASE_ARM
    prog.arm_unlocked = True
    prog.presence_prob = 0.0
  elif name == "box":
    prog.phase = PHASE_BOX
    prog.arm_unlocked = True
    prog.presence_prob = min(prog.presence_step, prog.presence_max)
  else:
    prog.phase = PHASE_BODY
    prog.arm_unlocked = False
    prog.presence_prob = 0.0
  prog.phase_iters = 0
  prog.hold_ok = 0
  prog.iters_since_bump = 0


def _sync_curriculum(prog, collector, trainer) -> None:
  collector.set_arm_unlocked(prog.arm_unlocked)
  collector.set_presence_prob(prog.presence_prob)
  trainer.arm_unlocked = bool(prog.arm_unlocked)


def run_real(args: argparse.Namespace) -> int:
  from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
  from homierl_mjlab.rl.stage3_bptt.collector import Stage3SimCollector
  from homierl_mjlab.rl.stage3_bptt.curriculum_progress import ProgressiveCurriculum
  from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer

  cur = args.curriculum.upper()
  if cur not in ("A", "B"):
    print(f"[warn] unknown curriculum {args.curriculum}; using A")
    cur = "A"
  use_auto = bool(args.curriculum_auto) and str(args.control_mode) == "body_ee_palm_delta"
  if bool(args.curriculum_auto) and not use_auto:
    print("[warn] curriculum-auto only applies to body_ee_palm_delta; using static A|B")
  print(
    f"[stage3_bptt] curriculum={cur} auto={use_auto} "
    f"(body→arm→box when auto; else static A/B)"
  )

  n_envs = int(args.num_envs)
  steps_per_env = max(1000, int(args.warmup_collect) + 64)
  need_cap = n_envs * steps_per_env
  if n_envs >= 512:
    print(
      f"[warn] num_envs={n_envs} → replay ~{need_cap} transitions "
      f"(SuperMimic-style {steps_per_env} steps/env); needs substantial VRAM."
    )

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_name = (args.run_name or "stage3_bptt").strip() or "stage3_bptt"
  log_dir = Path(args.log_dir) / f"{stamp}_{run_name}"
  log_dir.mkdir(parents=True, exist_ok=True)

  cfg = Stage3BpttCfg(
    device=args.device,
    stage2_checkpoint=args.stage2_checkpoint,
    log_dir=str(log_dir),
    curriculum=cur,
    curriculum_auto=use_auto,
    collect_steps_per_iter=args.collect_per_iter,
    replay_capacity=max(200_000, need_cap),
    control_mode=str(args.control_mode),
  )
  cfg.action_semantics = cfg.resolve_action_semantics()
  if args.w_teacher is not None:
    cfg.w_teacher = float(args.w_teacher)
  if args.w_loco is not None:
    cfg.w_loco = float(args.w_loco)
  if args.ee_delta_scale_m is not None:
    cfg.ee_delta_scale_m = float(args.ee_delta_scale_m)
  if args.body_trust_region is not None:
    cfg.body_trust_region = float(args.body_trust_region)
  if args.body_exec_mix is not None:
    cfg.body_exec_mix = float(args.body_exec_mix)
  print(
    f"[stage3_bptt] num_envs={n_envs} replay_capacity={cfg.replay_capacity} "
    f"(~{cfg.replay_capacity // n_envs} steps/env) "
    f"control_mode={cfg.control_mode} semantics={cfg.action_semantics}"
  )
  if cfg.control_mode == "body_ee_palm_delta":
    print(
      f"[stage3_bptt] body_ee: ee_delta_scale_m={cfg.ee_delta_scale_m} "
      f"w_teacher={cfg.w_teacher} w_loco={cfg.w_loco} "
      f"body_exec_teacher={cfg.body_exec_teacher} "
      f"body_exec_mix={cfg.body_exec_mix} "
      f"body_trust_region={cfg.body_trust_region} — FRESH log/replay "
      "(Phase-1/2 PPO rewards blended; trust-region stops takeoff)."
    )
  print(f"[INFO] Logging experiment in directory: {log_dir}")
  writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
  writer.add_text(
    "config/stage3_bptt",
    (
      f"ee_mode={args.ee_mode} control_mode={args.control_mode} "
      f"semantics={cfg.action_semantics} curriculum={cur} auto={use_auto} "
      f"start_phase={args.start_phase} num_envs={n_envs} "
      f"w_teacher={cfg.w_teacher} stage2={args.stage2_checkpoint}"
    ),
    0,
  )

  collector = Stage3SimCollector(
    cfg,
    stage2_checkpoint=args.stage2_checkpoint,
    num_envs=n_envs,
    ee_mode=str(args.ee_mode),
    actor=None,
  )
  control_fn, obs_fn, feat_fn, geom_fn, palm_step_fn = collector.make_fns()
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
    palm_step_fn=palm_step_fn,
    num_envs=int(collector.env.num_envs),
  )
  collector.actor = trainer.actor
  collector.ee_mode = str(args.ee_mode)

  prog = None
  if use_auto:
    prog = ProgressiveCurriculum.from_cfg(cfg)
    prog.max_ep_len = float(collector.max_episode_length_steps())
    _apply_start_phase(prog, args.start_phase)
    _sync_curriculum(prog, collector, trainer)
    print(
      f"[stage3_bptt] progressive phase={prog.phase_name} "
      f"arm_unlocked={prog.arm_unlocked} presence={prog.presence_prob:.2f} "
      f"max_ep_len≈{prog.max_ep_len:.0f} | gates: "
      f"body(mse≤{prog.teacher_mse_max}, ep≥{prog.ep_len_frac_min:.0%}, "
      f"≥{prog.body_min_iters}it) → "
      f"arm(reach≥{prog.reach_unlock_box}, ≥{prog.arm_min_iters}it) → "
      f"box(clear≥{prog.clear_reach_bump}, +{prog.presence_step}→{prog.presence_max})"
    )
  else:
    collector.set_arm_unlocked(True)
    trainer.arm_unlocked = True

  print(
    f"[stage3_bptt] ee_mode={collector.ee_mode} "
    f"actor_out_dim={trainer.actor_out_dim} obs_dim={collector.obs_dim} "
    f"arm_unlocked={collector.arm_unlocked}"
  )

  print(f"[info] warmup collect {args.warmup_collect} steps …")
  t0 = time.perf_counter()
  logs = collector.collect(trainer.replay, args.warmup_collect)
  warm_s = time.perf_counter() - t0
  if prog is not None:
    logs.update(prog.status_dict())
  print(json.dumps({**logs, "wall_s": warm_s}, indent=2))
  print(f"[info] warmup wall {warm_s:.1f}s")

  def _save_ckpt(tag: int | str) -> Path:
    ckpt = log_dir / f"model_{tag}.pt"
    payload = trainer.state_dict()
    payload["stage2_checkpoint"] = args.stage2_checkpoint
    payload["log_dir"] = str(log_dir)
    payload["control_mode"] = cfg.control_mode
    payload["action_semantics"] = cfg.action_semantics
    if prog is not None:
      payload["curriculum_progress"] = prog.state_dict()
    torch.save(payload, ckpt)
    print(f"[info] saved {ckpt}")
    return ckpt

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
    if prog is not None:
      before = prog.phase_name, prog.presence_prob
      clog_status = prog.step(merged)
      merged.update(clog_status)
      _sync_curriculum(prog, collector, trainer)
      for ev in prog.events:
        print(f"[curriculum] iter={it} {ev}")
      after = prog.phase_name, prog.presence_prob
      if after != before:
        _save_ckpt(f"curr_{prog.phase_name}_iter{it}")
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
    _tb_log(
      writer,
      it,
      merged,
      collect_s=collect_s,
      update_s=update_s,
      env_steps=env_steps,
      iter_s=iter_s,
    )
    if (it + 1) % save_every == 0:
      if any(
        isinstance(merged.get(k), float) and merged[k] != merged[k]
        for k in ("wm_loss", "critic_loss", "actor_loss")
      ):
        print(f"[warn] skip periodic save at iter {it+1}: non-finite train losses")
      else:
        _save_ckpt(it + 1)

  if tot_iter % save_every != 0:
    _save_ckpt(tot_iter)

  writer.close()
  print(f"[ok] real train loop finished (curriculum {cur} auto={use_auto}) log_dir={log_dir}")
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
