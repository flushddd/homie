"""Offline validity checks (no full GPU train required)."""

from __future__ import annotations

import torch

from homierl_mjlab.rl.stage3_bptt.dls import dls_toward, finite_diff_jacobian_check
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor, Stage3Critic, WorldModel
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import stage3_reward


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
  # Toy FK: ee = R @ q[:3] with fixed R
  R = torch.randn(3, 3)
  R = R / (R.norm() + 1e-6)

  def f_ee(q):
    return (R @ q[0, :3]).unsqueeze(0)

  q = torch.randn(1, 4)
  # Analytic J for first 3 cols = R, last = 0
  J = torch.zeros(1, 3, 4)
  J[0, :, :3] = R
  return finite_diff_jacobian_check(f_ee, q, J)


def check_bptt_actor_graph() -> dict[str, float]:
  """Tiny synthetic BPTT: actor → control → WM + palm_kin → reward → J."""
  device = torch.device("cpu")
  state_dim, ctrl_dim, obs_dim = 16, 8, 12
  actor = Stage3Actor(obs_dim, (32, 32), action_scale_m=0.02)
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
    err = action - palm
    n = torch.linalg.norm(err, dim=-1, keepdim=True).clamp_min(1e-9)
    return palm + err * (rate_m / n).clamp(max=1.0)

  s = torch.randn(8, state_dim, requires_grad=True)
  goal = torch.randn(8, 3)
  geom = torch.randn(8, 3)
  ctx = torch.cat((goal, geom), -1)
  last = torch.zeros(8, 3)
  rets = []
  disc = torch.ones(8)
  for _ in range(4):
    obs = torch.cat((s[..., :9], last), -1)
    a = actor(obs)
    u = torch.zeros(8, ctrl_dim)
    u = u.clone()
    u[:, :3] = a
    s_wm = wm(s, u, ctx)
    palm_kin = palm_step(s, a, goal, geom)
    s = torch.cat((s_wm[..., :-3], palm_kin), dim=-1)
    palm = s[..., -3:]
    clearance = torch.ones(8, 3) * 0.1
    r = stage3_reward(
      palm_pos=palm,
      goal_pos=goal,
      clearance=clearance,
      action=a,
      last_action=last,
    )
    rets.append(disc * r)
    disc = disc * 0.99
    last = a
  v = critic(torch.cat((s, ctx), -1))
  j = torch.stack(rets, 0).sum(0) + disc * v
  loss = -j.mean()
  loss.backward()
  gn = sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None)
  wm_changed = any(p.grad is not None for p in wm.parameters())
  return {
    "actor_grad_sum": float(gn),
    "wm_has_grad": float(wm_changed),
    "J_mean": float(j.detach().mean()),
  }


def run_all_checks() -> dict[str, object]:
  out: dict[str, object] = {}
  out["dls_grad"] = check_dls_grad()
  out["fd_jac_rel_err"] = check_fd_jacobian()
  out["bptt_graph"] = check_bptt_actor_graph()
  return out


if __name__ == "__main__":
  import json

  print(json.dumps(run_all_checks(), indent=2))
