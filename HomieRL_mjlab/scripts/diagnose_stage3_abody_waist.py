#!/usr/bin/env python3
"""Diagnose Stage3 BPTT squat/waist lean: grads, obs align, waist index.

Runs a short real collect + one actor update instrumentation. Prints facts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))


def _parse() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument(
    "--bptt-checkpoint",
    type=str,
    default="logs/rsl_rl/homie_g1/phase3/model_9000.pt",
  )
  p.add_argument("--stage2-checkpoint", type=str, default="")
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument("--collect-steps", type=int, default=64)
  return p.parse_args()


def _tensor_dict_norm(grads: dict[str, torch.Tensor]) -> float:
  sq = 0.0
  for g in grads.values():
    sq += float(g.detach().float().pow(2).sum().item())
  return sq**0.5


def _cosine(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
  keys = sorted(set(a) & set(b))
  if not keys:
    return float("nan")
  va = torch.cat([a[k].reshape(-1) for k in keys])
  vb = torch.cat([b[k].reshape(-1) for k in keys])
  if va.numel() == 0:
    return float("nan")
  return float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())


def main() -> int:
  args = _parse()
  ckpt_path = Path(args.bptt_checkpoint)
  if not ckpt_path.is_file():
    print(f"error: missing {ckpt_path}", file=sys.stderr)
    return 2
  payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  print(
    json.dumps(
      {
        "ckpt": str(ckpt_path),
        "action_semantics": payload.get("action_semantics"),
        "obs_dim": payload.get("obs_dim"),
        "loco_obs_dim": payload.get("loco_obs_dim"),
        "iter": payload.get("iter"),
        "arm_unlocked": payload.get("arm_unlocked"),
        "curriculum_progress": payload.get("curriculum_progress"),
        "cfg.teacher_action_norm": (payload.get("cfg") or {}).get(
          "teacher_action_norm"
        ),
        "cfg.body_exec_mix": (payload.get("cfg") or {}).get("body_exec_mix"),
        "cfg.body_trust_region": (payload.get("cfg") or {}).get(
          "body_trust_region"
        ),
        "cfg.w_teacher": (payload.get("cfg") or {}).get("w_teacher"),
      },
      indent=2,
    )
  )

  from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
  from homierl_mjlab.rl.stage3_bptt.collector import (
    Stage3SimCollector,
    normalize_teacher_actions,
  )
  from homierl_mjlab.rl.stage3_bptt.hierarchical import (
    inject_subgoal_into_grasp,
    palm_pos_world,
  )
  from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer

  saved = payload.get("cfg") or {}
  stage2 = args.stage2_checkpoint or str(payload.get("stage2_checkpoint") or "")
  for cand in (
    stage2,
    "logs/rsl_rl/homie_g1/phase2/model_18600.pt",
    str(_ROOT / "logs/rsl_rl/homie_g1/phase2/model_18600.pt"),
  ):
    if cand and Path(cand).is_file():
      stage2 = cand
      break
  if not Path(stage2).is_file():
    print(f"error: stage2 missing: {stage2}", file=sys.stderr)
    return 2

  cfg = Stage3BpttCfg(
    device=args.device,
    stage2_checkpoint=stage2,
    curriculum="A",
    curriculum_auto=False,
    body_exec_teacher=bool(saved.get("body_exec_teacher", True)),
    body_exec_mix=float(saved.get("body_exec_mix", 0.0)),
    body_trust_region=float(saved.get("body_trust_region", 0.2)),
    w_teacher=float(saved.get("w_teacher", 2.0)),
    teacher_action_norm=str(saved.get("teacher_action_norm", "none")),
    teacher_action_clip=float(saved.get("teacher_action_clip", 100.0)),
    body_action_clip=float(saved.get("body_action_clip", 100.0)),
    include_loco_obs=bool(saved.get("include_loco_obs", True)),
    loco_history_length=int(saved.get("loco_history_length", 6)),
    control_mode=str(payload.get("control_mode") or "body_ee_palm_delta"),
    imagine_horizon=int(saved.get("imagine_horizon", 8)),
    sdf_query_count=int(saved.get("sdf_query_count", 64)),
  )
  if "actor_hidden" in saved:
    cfg.actor_hidden = tuple(saved["actor_hidden"])
  cfg.action_semantics = cfg.resolve_action_semantics()

  collector = Stage3SimCollector(
    cfg,
    stage2_checkpoint=stage2,
    num_envs=int(args.num_envs),
    ee_mode="actor",
    actor=None,
    play=False,
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
    loco_obs_dim=int(collector.loco_obs_dim),
  )
  trainer.load_state_dict(payload)
  collector.actor = trainer.actor
  collector.set_arm_unlocked(bool(payload.get("arm_unlocked", True)))
  trainer.arm_unlocked = bool(payload.get("arm_unlocked", True))

  # --- waist action index ---
  jp = collector.env.action_manager.get_term("joint_pos")
  names = list(getattr(jp, "target_names", []) or [])
  waist_idx = names.index("waist_pitch_joint") if "waist_pitch_joint" in names else -1
  print(
    json.dumps(
      {
        "joint_pos.target_names": names,
        "waist_pitch_action_index": waist_idx,
        "loco_obs_dim_runtime": collector.loco_obs_dim,
        "loco_one_step_dim": collector.loco_one_step_dim,
        "loco_history_length": collector.loco_history_length,
        "obs_dim_runtime": collector.obs_dim,
        "env_action_dim": int(collector.env.action_manager.total_action_dim),
      },
      indent=2,
    )
  )

  # --- pre vs post inject loco obs / teacher ---
  collector._force_reach_flags()
  collector.palm_state.ensure_all(collector.env, arm_left=collector.arm_left)
  loco_pre = collector._current_loco_step()
  assert loco_pre is not None
  a_body_pre, a_ee = collector._actor_body_ee()
  p_sub = collector.palm_state.apply_palm_delta_base(
    collector.env,
    a_ee * 0.0,  # zero delta → p_sub≈palm; still writes grasp
    arm_left=collector.arm_left,
    rate_m=cfg.subgoal_rate_limit_m,
  )
  # Explicit non-palm inject to measure delta: push subgoal 5cm forward in world.
  p_sub2 = p_sub.clone()
  p_sub2[:, 0] = p_sub2[:, 0] + 0.05
  inject_subgoal_into_grasp(collector.env, p_sub2)
  loco_post = collector._current_loco_step()
  assert loco_post is not None
  # Advance student hist on post-inject (same path as collect) before teacher.
  if collector._loco_hist is not None:
    loco_hist_post = collector._advance_loco_hist(loco_post)
  else:
    loco_hist_post = None
  obs_td = collector.vec.get_observations()
  with torch.inference_mode():
    full_t = collector.policy(obs_td)
  full_t = normalize_teacher_actions(
    full_t,
    mode=str(cfg.teacher_action_norm),
    clip=float(cfg.teacher_action_clip),
  )
  teacher_post = full_t[..., : collector.body_dim]
  # Teacher on pre-inject: restore palm inject then query (advances teacher hist).
  inject_subgoal_into_grasp(collector.env, p_sub)
  with torch.inference_mode():
    full_pre = collector.policy(collector.vec.get_observations())
  full_pre = normalize_teacher_actions(
    full_pre, mode=str(cfg.teacher_action_norm), clip=float(cfg.teacher_action_clip)
  )
  teacher_pre = full_pre[..., : collector.body_dim]
  # Student body on stacked post-inject hist (BC path).
  if (
    loco_hist_post is not None
    and collector.actor is not None
    and getattr(collector.actor, "body_net", None) is not None
  ):
    with torch.no_grad():
      a_body_hist = collector.actor._clip_body(
        collector.actor.body_net(loco_hist_post)
      )
  else:
    a_body_hist = a_body_pre

  grasp = collector.env.command_manager.get_term("grasp")
  twist = collector.env.command_manager.get_term("twist")
  print(
    json.dumps(
      {
        "align.loco_pre_vs_post_l2": float(
          (loco_pre - loco_post).norm(dim=-1).mean().item()
        ),
        "align.loco_changed_frac": float(
          ((loco_pre - loco_post).abs().amax(dim=-1) > 1e-6).float().mean().item()
        ),
        "align.teacher_pre_vs_post_l2": float(
          (teacher_pre - teacher_post).norm(dim=-1).mean().item()
        ),
        "align.teacher_waist_pre": float(teacher_pre[0, waist_idx].item())
        if waist_idx >= 0
        else None,
        "align.teacher_waist_post": float(teacher_post[0, waist_idx].item())
        if waist_idx >= 0
        else None,
        "align.student_waist_on_pre_loco": float(a_body_pre[0, waist_idx].item())
        if waist_idx >= 0
        else None,
        "align.student_waist_on_hist6": float(a_body_hist[0, waist_idx].item())
        if waist_idx >= 0
        else None,
        "align.loco_hist_dim": int(loco_hist_post.shape[-1])
        if loco_hist_post is not None
        else 0,
        "flags.is_target": float(grasp.is_target[0].item()),
        "flags.is_waist": float(grasp.is_waist[0].item()),
        "flags.height_cmd": float(twist.command[0, 3].item())
        if twist.command.shape[-1] > 3
        else None,
        "flags.lin_vel_x": float(twist.command[0, 0].item()),
      },
      indent=2,
    )
  )

  # --- short collect into replay ---
  logs = collector.collect(trainer.replay, int(args.collect_steps))
  print("collect_logs", json.dumps({k: logs[k] for k in list(logs)[:20]}, indent=2))

  # Sample waist from last logged replay slots
  if trainer.replay.teacher_body is not None and trainer.replay.size > 0:
    n = min(32, trainer.replay.size)
    idx = torch.arange(n, device=trainer.device)
    tb = trainer.replay.teacher_body[idx]
    aa = trainer.replay.actor_action[idx] if trainer.replay.actor_action is not None else None
    lo = trainer.replay.loco_obs[idx] if trainer.replay.loco_obs is not None else None
    out = {
      "replay.n": n,
      "replay.teacher_waist_mean": float(tb[:, waist_idx].mean().item())
      if waist_idx >= 0
      else None,
      "replay.teacher_waist_std": float(tb[:, waist_idx].std().item())
      if waist_idx >= 0
      else None,
      "replay.teacher_body_absmax": float(tb.abs().max().item()),
      "replay.teacher_looks_tanh_sat": float(
        (tb.abs() > 0.99).float().mean().item()
      ),
    }
    if aa is not None and waist_idx >= 0:
      out["replay.student_waist_mean"] = float(aa[:, waist_idx].mean().item())
      out["replay.student_vs_teacher_waist_mse"] = float(
        F.mse_loss(aa[:, waist_idx], tb[:, waist_idx]).item()
      )
    if lo is not None and trainer.actor.body_net is not None:
      with torch.no_grad():
        ab, _ = trainer.actor(
          torch.cat(
            (
              lo,
              torch.zeros(
                n,
                collector.obs_dim - collector.loco_obs_dim,
                device=trainer.device,
              ),
            ),
            dim=-1,
          )
        )
      # Better: rebuild full obs via body path — use loco only for dual head.
      ab = trainer.actor._clip_body(trainer.actor.body_net(lo))
      out["replay.bc_on_stored_loco_waist_mse"] = float(
        F.mse_loss(ab[:, waist_idx], tb[:, waist_idx]).item()
      ) if waist_idx >= 0 else None
      out["replay.bc_on_stored_loco_body_mse"] = float(F.mse_loss(ab, tb).item())
    print(json.dumps(out, indent=2))

  # --- imag: a_body repetition + grad split ---
  windows = trainer.replay.sample_contiguous_windows(
    min(32, max(8, trainer.replay._num_complete_steps() - 2)),
    1,
    num_envs_hint=trainer.num_envs_hint,
  )
  (
    states,
    _c,
    _r,
    _t,
    _tr,
    goals,
    geoms,
    present,
    teacher_body,
    _actor_logged,
    loco_obs,
  ) = trainer._unpack_windows(windows)
  s0 = states[:, 0].detach()
  loco0 = loco_obs[:, 0].detach() if loco_obs is not None else None
  last_a = torch.zeros(s0.shape[0], trainer.actor_out_dim, device=trainer.device)
  bodies = []
  for t in range(int(cfg.imagine_horizon)):
    obs = obs_fn(s0, goals, geoms, last_a, loco_obs=loco0)
    a_body, a_ee = trainer.actor(obs)
    bodies.append(a_body.detach())
    last_a = torch.cat((a_body, a_ee), dim=-1).detach()
    # advance state lightly so reach changes; loco fixed
    u = control_fn(s0, last_a, goals, geoms)
    ctx = torch.cat((goals, geoms), dim=-1)
    with torch.no_grad():
      s0 = trainer.wm(s0, u, ctx)
  body_stack = torch.stack(bodies, dim=0)  # H,B,14
  print(
    json.dumps(
      {
        "imag.body_step_to_step_l2_mean": float(
          (body_stack[1:] - body_stack[:-1]).norm(dim=-1).mean().item()
        ),
        "imag.body_identical_all_H": bool(
          (body_stack - body_stack[:1]).abs().max().item() < 1e-7
        ),
        "imag.waist_over_H": [
          float(body_stack[t, 0, waist_idx].item()) for t in range(body_stack.shape[0])
        ]
        if waist_idx >= 0
        else None,
      },
      indent=2,
    )
  )

  # Gradients: BC-only vs BPTT-only vs combined on body_net
  def _body_grads(loss: torch.Tensor) -> dict[str, torch.Tensor]:
    trainer.opt_actor.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    out = {}
    for name, p in trainer.actor.named_parameters():
      if "body_net" not in name:
        continue
      if p.grad is None:
        out[name] = torch.zeros_like(p)
      else:
        out[name] = p.grad.detach().clone()
    return out

  # Rebuild one update-like forward for grads
  s = states[:, 0].detach().requires_grad_(True)
  ctx = torch.cat((goals, geoms), dim=-1)
  last_a = torch.zeros(s.shape[0], trainer.actor_out_dim, device=trainer.device)
  loco0 = loco_obs[:, 0].detach() if loco_obs is not None else None
  for p in trainer.wm.parameters():
    p.requires_grad_(False)
  for p in trainer.critic.parameters():
    p.requires_grad_(False)
  returns = []
  discount = torch.ones(s.shape[0], device=trainer.device)
  a0_body = None
  for _t in range(int(cfg.imagine_horizon)):
    obs = obs_fn(s, goals, geoms, last_a, loco_obs=loco0)
    a_body_t, a_ee_t = trainer.actor(obs)
    if not trainer.arm_unlocked:
      a_ee_t = torch.zeros_like(a_ee_t)
    a = torch.cat((a_body_t, a_ee_t), dim=-1)
    if a0_body is None:
      a0_body = a_body_t
    u = control_fn(s, a, goals, geoms)
    s_wm = trainer.wm(s, u, ctx)
    palm_kin = palm_step_fn(s, a, goals, geoms)
    s = torch.cat((s_wm[..., :-3], palm_kin), dim=-1)
    # imag reward: stability proxy only for grad presence
    from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import stage3_reward

    palm, clearance = geom_fn(s, goals, geoms)
    r, _ = stage3_reward(
      palm_pos=palm,
      goal_pos=goals[..., :3],
      clearance=clearance,
      action=a[..., -3:],
      last_action=last_a[..., -3:],
      w_target=0.0,
      w_approach=0.0,
      w_smooth=0.0,
      w_soft_success=0.0,
      return_terms=True,
    )
    returns.append(discount * r)
    discount = discount * cfg.gamma
    last_a = a
  j = torch.stack(returns, dim=0).sum(0)
  loss_bptt = -j.mean()
  tgt = teacher_body[:, 0].detach()
  loss_mse = F.mse_loss(a0_body, tgt)
  w_t = float(cfg.w_teacher)

  g_mse = _body_grads(loss_mse)
  g_bptt = _body_grads(loss_bptt)
  g_comb = _body_grads(loss_bptt + w_t * loss_mse)

  def _waist_out_grad(loss: torch.Tensor) -> float:
    trainer.opt_actor.zero_grad(set_to_none=True)
    # gradient of loss w.r.t. a0 waist via autograd on a0_body
    return float("nan")

  # Direct d(loss)/d(a0_body[waist])
  a0 = a0_body
  d_mse = torch.autograd.grad(loss_mse, a0, retain_graph=True, allow_unused=True)[0]
  d_bptt = torch.autograd.grad(loss_bptt, a0, retain_graph=True, allow_unused=True)[0]
  d_comb = torch.autograd.grad(
    loss_bptt + w_t * loss_mse, a0, retain_graph=True, allow_unused=True
  )[0]

  last = trainer.actor.body_net[-1]
  assert isinstance(last, torch.nn.Linear)
  trainer.opt_actor.zero_grad(set_to_none=True)
  (loss_bptt + w_t * loss_mse).backward()
  last_grad = last.weight.grad
  waist_row_norm = (
    float(last_grad[waist_idx].norm().item())
    if last_grad is not None and waist_idx >= 0
    else float("nan")
  )

  print(
    json.dumps(
      {
        "grad.teacher_mse": float(loss_mse.detach().item()),
        "grad.J_mean": float(j.detach().mean().item()),
        "grad.body_net_norm_mse": _tensor_dict_norm(g_mse),
        "grad.body_net_norm_bptt": _tensor_dict_norm(g_bptt),
        "grad.body_net_norm_comb": _tensor_dict_norm(g_comb),
        "grad.cosine_mse_vs_bptt": _cosine(g_mse, g_bptt),
        "grad.weighted_mse_norm": w_t * _tensor_dict_norm(g_mse),
        "grad.da0_waist_mse": float(d_mse[:, waist_idx].abs().mean().item())
        if d_mse is not None and waist_idx >= 0
        else None,
        "grad.da0_waist_bptt": float(d_bptt[:, waist_idx].abs().mean().item())
        if d_bptt is not None and waist_idx >= 0
        else None,
        "grad.da0_waist_comb": float(d_comb[:, waist_idx].abs().mean().item())
        if d_comb is not None and waist_idx >= 0
        else None,
        "grad.body_net_last_waist_row_norm": waist_row_norm,
        "grad.last_weight_finite": bool(
          last_grad is not None and torch.isfinite(last_grad).all().item()
        ),
      },
      indent=2,
    )
  )

  # nan_guard path check on a real update
  logs_u = trainer.update_actor_bptt()
  print("update_actor_bptt", json.dumps(logs_u, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
