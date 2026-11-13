"""Offline validity checks (no full GPU train required)."""

from __future__ import annotations

import torch

from homierl_mjlab.rl.stage3_bptt.cfg import ACTION_SEMANTICS_BODY_EE, Stage3BpttCfg
from homierl_mjlab.rl.stage3_bptt.dls import dls_toward, finite_diff_jacobian_check
from homierl_mjlab.rl.stage3_bptt.hierarchical import rate_limit
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor, Stage3Critic, WorldModel
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import stage3_reward
from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer


def check_dls_grad() -> dict[str, float]:
  b, n = 4, 4
  torch.manual_seed(0)
  J = torch.randn(b, 3, n, requires_grad=True)
  q = torch.randn(b, n, requires_grad=True)
  err = torch.randn(b, 3)
  q2 = dls_toward(J, q, err, iterations=2)
  loss = q2.pow(2).mean()
  loss.backward()
  return {
    "dq_grad_q": float(q.grad.abs().mean()),
    "dq_grad_J": float(J.grad.abs().mean()),
    "loss": float(loss.item()),
  }


def check_fd_jacobian() -> float:
  R = torch.randn(3, 3)
  R = R / (R.norm() + 1e-6)

  def f_ee(q):
    return (R @ q[0, :3]).unsqueeze(0)

  q = torch.randn(1, 4)
  J = torch.zeros(1, 3, 4)
  J[0, :, :3] = R
  return finite_diff_jacobian_check(f_ee, q, J)


def check_palm_delta_identity() -> dict[str, float]:
  """Fixed palm + delta → subgoal (base identity before rate-limit)."""
  palm_b = torch.tensor([[0.35, -0.12, 0.05]])
  delta_b = torch.tensor([[0.02, -0.01, 0.03]])
  p_sub_b = palm_b + delta_b
  err = float((p_sub_b - (palm_b + delta_b)).abs().max().item())
  # Rate-limit from prev=palm should allow full step if |delta| < rate.
  prev_w = palm_b.clone()
  raw_w = p_sub_b.clone()
  limited = rate_limit(raw_w, prev_w, max_step_m=0.04)
  step_n = float((limited - prev_w).norm(dim=-1).item())
  return {
    "identity_err": err,
    "rate_limited_step_m": step_n,
    "delta_norm": float(delta_b.norm().item()),
  }


def check_body_ee_actor_grads() -> dict[str, float]:
  """Body and EE heads both receive nonzero grads; teacher MSE path works."""
  obs_dim = 3 + 3 + 3 + 3 + 17 + 8 + 6  # sdf_k=8 toy
  actor = Stage3Actor(
    obs_dim,
    (64, 64),
    body_dim=14,
    ee_dim=3,
    ee_delta_scale_m=0.03,
    control_mode="body_ee_palm_delta",
  )
  obs = torch.randn(4, obs_dim)
  a_body, a_ee = actor(obs)
  teacher = torch.randn_like(a_body)
  loss = (a_body - teacher).pow(2).mean() + a_ee.pow(2).mean()
  loss.backward()
  body_g = 0.0
  ee_g = 0.0
  last = actor.net[-1]
  assert isinstance(last, torch.nn.Linear) and last.weight.grad is not None
  body_g = float(last.weight.grad[:14].abs().sum().item())
  ee_g = float(last.weight.grad[14:].abs().sum().item())
  return {
    "body_grad_sum": body_g,
    "ee_grad_sum": ee_g,
    "a_body_norm": float(a_body.detach().norm(dim=-1).mean().item()),
    "a_ee_norm": float(a_ee.detach().norm(dim=-1).mean().item()),
    "out_dim": float(actor.out_dim),
  }


def check_loco_hist6_dual_head() -> dict[str, float]:
  """body_net takes flat 6×one_step; ee_net takes reach; grads both flow."""
  one, hist = 16, 6
  loco_dim = one * hist
  reach_dim = 3 + 3 + 3 + 3 + 17 + 4 + 6
  obs_dim = loco_dim + reach_dim
  actor = Stage3Actor(
    obs_dim,
    (32, 32),
    body_dim=14,
    ee_dim=3,
    ee_delta_scale_m=0.0,  # no hard-clip so EE grads are not killed
    control_mode="body_ee_palm_delta",
    loco_obs_dim=loco_dim,
    near_zero_body_init=False,
  )
  assert actor.body_net is not None and actor.ee_net is not None
  torch.manual_seed(0)
  obs = torch.randn(4, obs_dim)
  a_body, a_ee = actor(obs)
  teacher = torch.randn_like(a_body)
  ee_tgt = torch.randn_like(a_ee)
  loss = (a_body - teacher).pow(2).mean() + (a_ee - ee_tgt).pow(2).mean()
  loss.backward()
  assert isinstance(actor.body_net[-1], torch.nn.Linear)
  assert isinstance(actor.ee_net[-1], torch.nn.Linear)
  body_g = float(actor.body_net[-1].weight.grad.abs().sum().item())  # type: ignore[union-attr]
  ee_g = float(actor.ee_net[-1].weight.grad.abs().sum().item())  # type: ignore[union-attr]
  first = actor.body_net[0]
  assert isinstance(first, torch.nn.Linear)
  return {
    "loco_obs_dim": float(loco_dim),
    "body_grad_sum": body_g,
    "ee_grad_sum": ee_g,
    "a_body_norm": float(a_body.detach().norm(dim=-1).mean().item()),
    "in_features_body": float(first.in_features),
  }


def check_splice_last_hist_frame() -> dict[str, float]:
  """Only the last one-step frame is refreshed from predicted state."""
  from homierl_mjlab.rl.stage3_bptt.env_bridge import splice_loco_proprio_from_state

  one, hist = 80, 6
  loco_dim = one * hist
  loco = torch.randn(2, loco_dim)
  prefix = loco[..., :-one].clone()
  state = torch.randn(2, 40 + 13)  # n_joints=20 + extras with quat/ang/palm
  # Pack-like: q(20)|qd(20)|…|quat at -13:-9 style not required; splice uses
  # ang=-6:-3, quat via helper — use n_joints=20.
  n_j = 20
  default_q = torch.zeros(n_j)
  out = splice_loco_proprio_from_state(
    loco,
    state,
    n_joints=n_j,
    default_q=default_q,
    loco_one_step_dim=one,
  )
  prefix_ok = float((out[..., :-one] - prefix).abs().max().item())
  last_changed = float((out[..., -one:] - loco[..., -one:]).abs().max().item())
  return {
    "prefix_max_delta": prefix_ok,
    "last_max_delta": last_changed,
    "loco_dim": float(loco_dim),
  }


def check_refuse_old_semantics() -> dict[str, float]:
  """Trainer refuses silent load of old 3-D hier ckpts into body_ee."""
  cfg = Stage3BpttCfg(
    device="cpu",
    control_mode="body_ee_palm_delta",
    wm_batch=8,
    wm_window=2,
    imagine_horizon=2,
    replay_capacity=128,
  )
  state_dim, control_dim = 20, 10
  goal_dim, geom_dim = 3, 4
  obs_dim = 3 + 3 + 3 + 3 + 17 + 4 + 6
  feat_dim = state_dim + goal_dim + geom_dim

  def control_fn(state, action, goal, geom):
    u = torch.zeros(state.shape[0], control_dim, device=state.device)
    u[:, :3] = action[..., -3:]
    return u

  def obs_fn(state, goal, geom, last_a):
    b = state.shape[0]
    pad = torch.zeros(b, obs_dim, device=state.device)
    n = min(6, state.shape[-1])
    pad[:, :n] = state[:, :n]
    pad[:, 6:9] = goal
    la = min(last_a.shape[-1], 17)
    pad[:, 12 : 12 + la] = last_a[:, :la]
    return pad

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
  # Fake legacy ckpt (no action_semantics, 3-D actor weights mismatch).
  legacy = {
    "actor": Stage3Actor(obs_dim, (32,), control_mode="hierarchical_subgoal").state_dict(),
    "critic": trainer.critic.state_dict(),
    "critic_tgt": trainer.critic_tgt.state_dict(),
    "wm": trainer.wm.state_dict(),
    "opt_actor": trainer.opt_actor.state_dict(),
    "opt_critic": trainer.opt_critic.state_dict(),
    "opt_wm": trainer.opt_wm.state_dict(),
    "reward_schema": "stage3_v2_present_approach_stability",
  }
  refused = 0.0
  try:
    trainer.load_state_dict(legacy)
  except RuntimeError as e:
    refused = 1.0 if "action_semantics" in str(e) else 0.5
  return {
    "refused_legacy": refused,
    "expected_semantics": (
      1.0 if trainer.action_semantics == ACTION_SEMANTICS_BODY_EE else 0.0
    ),
  }


def check_bptt_actor_graph() -> dict[str, float]:
  """Tiny synthetic BPTT: body+ee actor → control → WM + palm_kin → reward → J."""
  device = torch.device("cpu")
  state_dim, ctrl_dim = 16, 8
  body_dim, ee_dim = 14, 3
  # Match body_ee obs packing: palm3+goal3+(g-p)3+prev3+last_a+sdf4+clr6
  sdf_k = 4
  obs_dim = 3 + 3 + 3 + 3 + body_dim + ee_dim + sdf_k + 6
  actor = Stage3Actor(
    obs_dim,
    (32, 32),
    body_dim=body_dim,
    ee_dim=ee_dim,
    ee_delta_scale_m=0.03,
    control_mode="body_ee_palm_delta",
  )
  wm = WorldModel(state_dim, ctrl_dim, ctx_dim=6, hidden=(64, 64))
  critic = Stage3Critic(state_dim + 6, (32, 32))
  for p in wm.parameters():
    p.requires_grad_(False)
  for p in critic.parameters():
    p.requires_grad_(False)

  rate_m = 0.04

  def palm_step(state, action, goal, geom):
    del goal, geom
    palm = state[..., -3:]
    delta = action[..., -3:]
    return palm + torch.clamp(delta, -rate_m, rate_m)

  s = torch.randn(8, state_dim, requires_grad=True)
  goal = torch.randn(8, 3)
  geom = torch.randn(8, 3)
  ctx = torch.cat((goal, geom), -1)
  last = torch.zeros(8, body_dim + ee_dim)
  rets = []
  disc = torch.ones(8)
  a0_body = None
  for _ in range(4):
    palm = s[..., -3:]
    zero = torch.zeros_like(palm)
    sdf = torch.zeros(8, sdf_k, device=s.device)
    clr = torch.ones(8, 6, device=s.device) * 0.1
    obs = torch.cat((palm, goal, goal - palm, zero, last, sdf, clr), -1)
    a_body, a_ee = actor(obs)
    if a0_body is None:
      a0_body = a_body
    a = torch.cat((a_body, a_ee), dim=-1)
    u = torch.zeros(8, ctrl_dim)
    u = u.clone()
    u[:, :3] = a_ee
    s_wm = wm(s, u, ctx)
    palm_kin = palm_step(s, a, goal, geom)
    s = torch.cat((s_wm[..., :-3], palm_kin), dim=-1)
    palm = s[..., -3:]
    clearance = torch.ones(8, 3) * 0.1
    r = stage3_reward(
      palm_pos=palm,
      goal_pos=goal,
      clearance=clearance,
      action=a_ee,
      last_action=last[..., -3:],
      present=torch.ones(8),
    )
    rets.append(disc * r)
    disc = disc * 0.99
    last = a
  v = critic(torch.cat((s, ctx), -1))
  j = torch.stack(rets, 0).sum(0) + disc * v
  teacher = torch.zeros_like(a0_body)
  loss = -j.mean() + 5.0 * (a0_body - teacher).pow(2).mean()
  loss.backward()
  last_lin = actor.net[-1]
  assert isinstance(last_lin, torch.nn.Linear) and last_lin.weight.grad is not None
  body_g = float(last_lin.weight.grad[:body_dim].abs().sum().item())
  ee_g = float(last_lin.weight.grad[body_dim:].abs().sum().item())
  gn = sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None)
  wm_changed = any(p.grad is not None for p in wm.parameters())
  return {
    "actor_grad_sum": float(gn),
    "body_grad_sum": body_g,
    "ee_grad_sum": ee_g,
    "wm_has_grad": float(wm_changed),
    "J_mean": float(j.detach().mean()),
  }


def check_present_reward_gate() -> dict[str, float]:
  palm = torch.zeros(4, 3)
  goal = torch.tensor([[0.3, 0.0, 0.0]]).expand(4, 3)
  clr = torch.full((4, 3), -0.2)
  a = torch.zeros(4, 3)
  last = torch.zeros(4, 3)
  r0, t0 = stage3_reward(
    palm_pos=palm,
    goal_pos=goal,
    clearance=clr,
    action=a,
    last_action=last,
    present=torch.zeros(4),
    return_terms=True,
  )
  r1, t1 = stage3_reward(
    palm_pos=palm,
    goal_pos=goal,
    clearance=clr,
    action=a,
    last_action=last,
    present=torch.ones(4),
    return_terms=True,
  )
  return {
    "w_safe_absent": float(t0["w_safe"].mean()),
    "w_margin_absent": float(t0["w_margin"].mean()),
    "w_safe_present": float(t1["w_safe"].mean()),
    "r_absent": float(r0.mean()),
    "r_present": float(r1.mean()),
  }


def check_progressive_curriculum() -> dict[str, float]:
  from homierl_mjlab.rl.stage3_bptt.curriculum_progress import (
    PHASE_ARM,
    PHASE_BODY,
    PHASE_BOX,
    ProgressiveCurriculum,
  )

  prog = ProgressiveCurriculum(
    body_min_iters=2,
    body_hold_iters=2,
    arm_min_iters=2,
    arm_hold_iters=2,
    box_min_iters_between_bumps=1,
    box_hold_iters=2,
    teacher_mse_max=0.1,
    ep_len_frac_min=0.5,
    reach_unlock_box=0.4,
    clear_reach_bump=0.2,
    presence_step=0.1,
    presence_max=0.3,
    ema_beta=0.0,  # use raw metrics
    max_ep_len=100.0,
  )
  assert prog.phase == PHASE_BODY and not prog.arm_unlocked
  # Not ready yet (mse high).
  for _ in range(3):
    prog.step(
      {
        "teacher_mse_collect": 0.5,
        "mean_episode_length": 90.0,
        "reach_success_frac": 0.0,
        "clear_reach_success_frac": 0.0,
      }
    )
  assert prog.phase == PHASE_BODY
  # Unlock arm.
  for _ in range(4):
    prog.step(
      {
        "teacher_mse_collect": 0.01,
        "mean_episode_length": 90.0,
        "reach_success_frac": 0.1,
        "clear_reach_success_frac": 0.0,
      }
    )
  assert prog.phase == PHASE_ARM and prog.arm_unlocked
  # Enter box.
  for _ in range(6):
    prog.step(
      {
        "teacher_mse_collect": 0.01,
        "mean_episode_length": 90.0,
        "reach_success_frac": 0.6,
        "clear_reach_success_frac": 0.1,
      }
    )
  assert prog.phase == PHASE_BOX
  assert prog.presence_prob > 0.0
  # Bump presence.
  for _ in range(6):
    prog.step(
      {
        "teacher_mse_collect": 0.01,
        "mean_episode_length": 90.0,
        "reach_success_frac": 0.6,
        "clear_reach_success_frac": 0.5,
      }
    )
  assert prog.presence_prob >= 0.2 - 1e-9
  return {
    "final_phase": float(prog.phase),
    "final_presence": float(prog.presence_prob),
    "arm_unlocked": 1.0 if prog.arm_unlocked else 0.0,
  }


def run_all_checks() -> dict[str, object]:
  out: dict[str, object] = {}
  out["dls_grad"] = check_dls_grad()
  out["fd_jac_rel_err"] = check_fd_jacobian()
  out["palm_delta"] = check_palm_delta_identity()
  body_ee = check_body_ee_actor_grads()
  out["body_ee_grads"] = body_ee
  assert body_ee["body_grad_sum"] > 0.0
  assert body_ee["ee_grad_sum"] > 0.0
  refuse = check_refuse_old_semantics()
  out["refuse_old_semantics"] = refuse
  assert refuse["refused_legacy"] >= 1.0
  hist6 = check_loco_hist6_dual_head()
  out["loco_hist6_dual_head"] = hist6
  assert hist6["body_grad_sum"] > 0.0
  assert hist6["ee_grad_sum"] > 0.0
  assert hist6["in_features_body"] == hist6["loco_obs_dim"]
  splice = check_splice_last_hist_frame()
  out["splice_last_hist_frame"] = splice
  assert splice["prefix_max_delta"] < 1e-6
  assert splice["last_max_delta"] > 1e-6
  out["bptt_graph"] = check_bptt_actor_graph()
  gate = check_present_reward_gate()
  out["present_gate"] = gate
  assert abs(gate["w_safe_absent"]) < 1e-8
  assert abs(gate["w_margin_absent"]) < 1e-8
  assert gate["w_safe_present"] < -1e-6
  g = out["bptt_graph"]
  assert g["actor_grad_sum"] > 0.0
  assert g["body_grad_sum"] > 0.0
  assert g["ee_grad_sum"] > 0.0
  assert g["wm_has_grad"] == 0.0
  curr = check_progressive_curriculum()
  out["progressive_curriculum"] = curr
  assert curr["arm_unlocked"] == 1.0
  assert curr["final_phase"] >= 2.0
  return out


if __name__ == "__main__":
  import json

  print(json.dumps(run_all_checks(), indent=2))
