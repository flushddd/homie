#!/usr/bin/env python3
"""Viser play for Stage-3 BPTT: body+EE actor or legacy hierarchical.

Draws two world spheres each frame:
  - orange/red: true task goal ``p_goal``
  - cyan: policy next-frame target (``p_sub`` = palm+delta or hier offset)

Examples::

  .venv/bin/python scripts/play_stage3_bptt.py \\
    --stage2-checkpoint logs/rsl_rl/homie_g1/phase2/model_18600.pt \\
    --bptt-checkpoint logs/.../model_200.pt --ee-mode actor
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))

_COLOR_GOAL = (1.0, 0.25, 0.15, 0.85)
_COLOR_SUB = (0.1, 0.85, 1.0, 0.95)


def _parse() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Stage-3 BPTT Viser play")
  p.add_argument(
    "--stage2-checkpoint",
    type=str,
    default="",
    help="Frozen phase-2/loco ckpt (falls back to payload inside --bptt-checkpoint)",
  )
  p.add_argument(
    "--bptt-checkpoint",
    type=str,
    default="",
    help="Stage3 BPTT ckpt (required for --ee-mode actor/attractor; optional for goal/palm)",
  )
  p.add_argument("--device", type=str, default="cuda:0")
  p.add_argument(
    "--curriculum",
    type=str,
    default="A",
    help="A=no crate (default play); B=crates like train",
  )
  p.add_argument(
    "--control-mode",
    type=str,
    default="",
    choices=("", "body_ee_palm_delta", "hierarchical_subgoal", "ee_residual"),
    help="Override ckpt/cfg control mode (empty = ckpt or body_ee default)",
  )
  p.add_argument(
    "--ee-mode",
    type=str,
    default="goal",
    choices=("actor", "attractor", "goal", "palm"),
    help=(
      "goal: inject true p_goal only (default; test loco+IK, no actor); "
      "palm: hold p_sub at palm (stand); "
      "attractor: heuristic toward p_goal; "
      "actor: BPTT policy"
    ),
  )
  p.add_argument(
    "--body-only",
    action="store_true",
    help=(
      "Shorthand: --zero-ee-delta --subgoal palm + arm chase OFF "
      "(a_body / lower-limb stability only)."
    ),
  )
  p.add_argument(
    "--zero-ee-delta",
    action="store_true",
    help=(
      "Force a_ee:=0 (ignore Stage3 EE head). Use with --subgoal palm|goal. "
      "Implies zero EE when --body-only."
    ),
  )
  p.add_argument(
    "--subgoal",
    type=str,
    default="auto",
    choices=("auto", "palm", "goal"),
    help=(
      "Where p_sub goes when a_ee is forced to 0: "
      "palm (hold current palm; loco-only), goal (inject Viser p_goal; DLS chase). "
      "auto: palm if --body-only else goal if --zero-ee-delta else unused."
    ),
  )
  p.add_argument(
    "--loco-source",
    type=str,
    default="actor",
    choices=("actor", "teacher", "zero"),
    help="Who fills a_body(14) in actor/body-only play: actor|teacher|zero.",
  )
  p.add_argument("--num-envs", type=int, default=1)
  p.add_argument(
    "--log-every",
    type=int,
    default=25,
    help="Print actor / p_sub / p_goal every N control steps (0=off)",
  )
  return p.parse_args()


def _install_dual_target_debug_vis(env, *, show_sub: bool) -> None:
  """Replace grasp debug spheres with p_goal (orange) + policy target (cyan)."""
  grasp = env.command_manager.get_term("grasp")

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    goal_w = getattr(env, "_homie_bptt_goal_w", None)
    sub_w = getattr(env, "_homie_bptt_sub_w", None)
    is_w = self.is_waist.detach().cpu().numpy().reshape(-1)
    for batch in env_indices:
      if is_w[batch] < 0.5:
        continue
      if goal_w is not None:
        g = goal_w[batch].detach().cpu().numpy()
        if np.linalg.norm(g) > 1e-6:
          visualizer.add_sphere(
            g,
            radius=0.045,
            color=_COLOR_GOAL,
            label=f"p_goal_{batch}",
          )
      if sub_w is not None:
        s = sub_w[batch].detach().cpu().numpy()
        if np.linalg.norm(s) > 1e-6:
          visualizer.add_sphere(
            s,
            radius=0.035,
            color=_COLOR_SUB,
            label=f"p_sub_{batch}",
          )
      elif not show_sub:
        center = self.target_pos_w[batch].detach().cpu().numpy()
        if np.linalg.norm(center) > 1e-6:
          visualizer.add_sphere(
            center,
            radius=0.035,
            color=_COLOR_SUB,
            label=f"p_cmd_{batch}",
          )

  grasp._debug_vis_impl = types.MethodType(_debug_vis_impl, grasp)
  grasp.cfg.debug_vis = True


def _residual_cmd_world(env, delta_b: torch.Tensor, *, arm_left: bool) -> torch.Tensor:
  from homierl_mjlab.mdp.base_frame import heading_quat
  from homierl_mjlab.rl.stage3_bptt.hierarchical import palm_pos_world
  from mjlab.utils.lab_api.math import quat_apply

  palm_w = palm_pos_world(env, arm_left=arm_left)
  heading = heading_quat(env.scene["robot"])
  delta_w = quat_apply(heading, delta_b)
  return palm_w + delta_w


def _body_target_to_world(robot, body_xyz: torch.Tensor, *, base_height: float) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply, yaw_quat

  root = robot.data.root_link_pos_w[:, :3]
  heading = yaw_quat(robot.data.root_link_quat_w)
  delta = quat_apply(heading, body_xyz)
  out = torch.empty_like(body_xyz)
  out[:, :2] = root[:, :2] + delta[:, :2]
  out[:, 2] = float(base_height) + body_xyz[:, 2]
  return out


def _goal_state(collector):
  """Active true-goal / subgoal buffers (palm_state or hier)."""
  if getattr(collector, "body_ee", False):
    return collector.palm_state
  return collector.hier


def _sync_true_goal_from_viser(collector) -> bool:
  """Refresh true ``p_goal_w`` from Viser slider desired body xyz."""
  grasp = collector.env.command_manager.get_term("grasp")
  if not bool(getattr(grasp, "_gui_enabled", None) and grasp._gui_enabled.value):
    return False
  if len(getattr(grasp, "_gui_sliders", [])) < 3:
    return False
  n = collector.env.num_envs
  body = torch.zeros(n, 3, device=collector.device)
  x = float(grasp._gui_sliders[0].value)
  y = float(grasp._gui_sliders[1].value)
  z = float(grasp._gui_sliders[2].value)
  body[:, 0] = x
  body[:, 1] = y
  body[:, 2] = z
  base_h = float(grasp.cfg.base_height_target)
  goal_w = _body_target_to_world(
    collector.env.scene["robot"], body, base_height=base_h
  )
  st = _goal_state(collector)
  st.p_goal_w = goal_w
  collector.env._homie_bptt_goal_body = body

  want_left = y > 0.0
  grasp.target_arm[:] = 1.0 if want_left else 2.0
  if bool(want_left) != bool(collector.arm_left):
    trained = str(getattr(collector.cfg, "arm_side", "right"))
    if not getattr(collector, "_arm_side_warned", False):
      print(
        f"[bptt-play] WARN: Viser y={y:+.3f} → "
        f"{'left' if want_left else 'right'} arm, but ckpt trained "
        f"arm_side={trained!r}. Re-anchoring palm; actor may be OOD."
      )
      collector._arm_side_warned = True
    collector.arm_left = bool(want_left)
    st.initialized[:] = False
    st.ensure_all(collector.env, arm_left=collector.arm_left)
    st.p_goal_w = _body_target_to_world(
      collector.env.scene["robot"], body, base_height=base_h
    )
  return True


class _PlayEnvAdapter:
  """Thin adapter so ViserPlayViewer can step while we reset subgoal state."""

  def __init__(self, collector):
    self.collector = collector
    self.vec = collector.vec

  @property
  def unwrapped(self):
    return self.vec.unwrapped

  @property
  def num_envs(self) -> int:
    return int(self.vec.num_envs)

  @property
  def device(self):
    return self.vec.device

  @property
  def cfg(self):
    return self.vec.cfg

  def close(self) -> None:
    return self.vec.close()

  def get_observations(self):
    return self.vec.get_observations()

  def __getattr__(self, name: str):
    return getattr(self.vec, name)

  def reset(self):
    out = self.vec.reset()
    c = self.collector
    st = _goal_state(c)
    st.initialized[:] = False
    st.ensure_all(c.env, arm_left=c.arm_left)
    c.last_a.zero_()
    c._force_reach_flags()
    self._publish_markers()
    return out

  def step(self, actions):
    obs, rew, dones, extras = self.vec.step(actions)
    c = self.collector
    c._force_reach_flags()
    done_mask = (
      dones.bool()
      if torch.is_tensor(dones)
      else torch.tensor(dones, device=c.device).bool()
    )
    if bool(done_mask.any()):
      done_ids = done_mask.nonzero(as_tuple=False).flatten()
      c.last_a[done_mask] = 0.0
      st = _goal_state(c)
      st.reset_envs(c.env, done_ids, arm_left=c.arm_left)
      if hasattr(c.policy, "reset"):
        c.policy.reset()
    self._publish_markers()
    return obs, rew, dones, extras

  def _publish_markers(self) -> None:
    c = self.collector
    env = c.env
    if c.body_ee or c.hierarchical:
      st = _goal_state(c)
      env._homie_bptt_goal_w = st.p_goal_w
      env._homie_bptt_sub_w = st.p_sub_w
    else:
      grasp = env.command_manager.get_term("grasp")
      env._homie_bptt_goal_w = grasp.target_pos_w
      env._homie_bptt_sub_w = getattr(env, "_homie_bptt_sub_w", None)


class Stage3BpttPlayPolicy:
  """PolicyProtocol: high-level BPTT action → stitch with frozen loco/teacher."""

  def __init__(
    self,
    collector,
    *,
    log_every: int = 25,
    body_only: bool = False,
    zero_ee: bool = False,
    subgoal: str = "palm",
    loco_source: str = "actor",
  ):
    self.collector = collector
    self.log_every = max(0, int(log_every))
    self.body_only = bool(body_only)
    self.zero_ee = bool(zero_ee) or self.body_only
    self.subgoal = str(subgoal)
    self.loco_source = str(loco_source)
    self._step = 0

  def reset(self) -> None:
    c = self.collector
    st = _goal_state(c)
    st.initialized[:] = False
    st.ensure_all(c.env, arm_left=c.arm_left)
    c.last_a.zero_()
    c._force_reach_flags()
    self._step = 0

  def _maybe_log(self, *, mode: str, packed_a: torch.Tensor) -> None:
    if self.log_every <= 0 or self._step % self.log_every != 0:
      return
    c = self.collector
    st = _goal_state(c)
    goal = st.p_goal_w[0].detach().float().cpu().tolist()
    sub = st.p_sub_w[0].detach().float().cpu().tolist()
    body = getattr(c.env, "_homie_bptt_goal_body", None)
    body_s = (
      f" body=({body[0,0]:.3f},{body[0,1]:.3f},{body[0,2]:.3f})"
      if body is not None
      else ""
    )
    if c.body_ee:
      a = packed_a[0].detach().float().cpu()
      bn = float(a[:14].norm().item())
      ee = a[14:17].tolist()
      print(
        f"[bptt-play] step={self._step} mode={mode} arm="
        f"{'L' if c.arm_left else 'R'} "
        f"a_body_norm={bn:.3f} "
        f"a_ee=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f}) "
        f"p_sub=({sub[0]:+.3f},{sub[1]:+.3f},{sub[2]:+.3f}) "
        f"p_goal=({goal[0]:+.3f},{goal[1]:+.3f},{goal[2]:+.3f})"
        f"{body_s}"
      )
    else:
      a = packed_a[0].detach().float().cpu().tolist()
      print(
        f"[bptt-play] step={self._step} mode={mode} "
        f"actor_a=({a[0]:+.4f},{a[1]:+.4f},{a[2]:+.4f}) "
        f"p_sub=({sub[0]:+.3f},{sub[1]:+.3f},{sub[2]:+.3f}) "
        f"p_goal=({goal[0]:+.3f},{goal[1]:+.3f},{goal[2]:+.3f})"
        f"{body_s}"
      )

  def __call__(self, obs_td):
    c = self.collector
    c._force_reach_flags()
    st = _goal_state(c)
    st.ensure_all(c.env, arm_left=c.arm_left)
    _sync_true_goal_from_viser(c)
    n = c.env.num_envs
    mode = str(getattr(c, "ee_mode", "actor"))

    from homierl_mjlab.rl.stage3_bptt.hierarchical import (
      inject_subgoal_into_grasp,
      palm_pos_world,
    )

    if c.body_ee:
      # Stage3 dual-head: a_body(14) + a_ee(3). Env residual held at 0;
      # arms chase injected p_sub via DLS when arm chase is ON.
      need_teacher_body = False
      if mode == "goal":
        p_sub = st.p_goal_w
        st.p_sub_w = p_sub.clone()
        a_ee = torch.zeros(n, 3, device=c.device)
        need_teacher_body = True
      elif mode == "palm":
        palm_w = palm_pos_world(c.env, arm_left=c.arm_left)
        st.p_sub_w = palm_w.clone()
        p_sub = st.p_sub_w
        a_ee = torch.zeros(n, 3, device=c.device)
        need_teacher_body = True
      elif mode == "actor" and (
        c.actor is not None or self.loco_source in ("teacher", "zero")
      ):
        if c.actor is not None:
          a_body, a_ee = c._actor_body_ee()
        else:
          a_body = torch.zeros(n, c.body_dim, device=c.device)
          a_ee = torch.zeros(n, 3, device=c.device)
        if self.loco_source == "zero":
          a_body = torch.zeros(n, c.body_dim, device=c.device)
        elif self.loco_source == "teacher":
          need_teacher_body = True
        # Force a_ee:=0 for body-only / zero-ee / non-actor loco — then pin
        # p_sub to palm (loco stability) or Viser p_goal (body + DLS).
        force_zero_ee = (
          self.zero_ee or self.body_only or self.loco_source != "actor"
        )
        if force_zero_ee:
          a_ee = torch.zeros(n, 3, device=c.device)
          st.last_delta_b.zero_()
          if self.subgoal == "goal":
            p_sub = st.p_goal_w
            st.p_sub_w = p_sub.clone()
          else:
            palm_w = palm_pos_world(c.env, arm_left=c.arm_left)
            st.p_sub_w = palm_w.clone()
            p_sub = st.p_sub_w
        else:
          p_sub = st.apply_palm_delta_base(
            c.env,
            a_ee,
            arm_left=c.arm_left,
            rate_m=c.cfg.subgoal_rate_limit_m,
          )
      else:
        from homierl_mjlab.rl.stage3_bptt.hierarchical import (
          attractor_palm_delta_b,
        )

        a_ee = attractor_palm_delta_b(
          c.env,
          st,
          arm_left=c.arm_left,
          scale_m=min(c.cfg.ee_delta_scale_m, 0.02),
        )
        p_sub = st.apply_palm_delta_base(
          c.env,
          a_ee,
          arm_left=c.arm_left,
          rate_m=c.cfg.subgoal_rate_limit_m,
        )
        need_teacher_body = True

      inject_subgoal_into_grasp(c.env, p_sub)
      c.env._homie_bptt_goal_w = st.p_goal_w
      c.env._homie_bptt_sub_w = st.p_sub_w
      obs_td = c.vec.get_observations()
      with torch.inference_mode():
        full_teacher = c.policy(obs_td)
      from homierl_mjlab.rl.stage3_bptt.collector import (
        normalize_teacher_actions,
        safe_exec_body,
      )

      # Same as phase-1/2: raw HIM mean, ±clip_actions.
      full_teacher = normalize_teacher_actions(
        full_teacher,
        mode=str(getattr(c.cfg, "teacher_action_norm", "none")),
        clip=float(getattr(c.cfg, "teacher_action_clip", 100.0)),
      )
      teacher_body = full_teacher[..., : c.body_dim].detach()
      if need_teacher_body or c.actor is None:
        a_body = teacher_body
        exec_body = teacher_body
      elif self.loco_source == "zero":
        a_body = torch.zeros(n, c.body_dim, device=c.device)
        exec_body = teacher_body
      else:
        # Trust-region in the normalized space — no takeoff.
        exec_body = safe_exec_body(
          a_body,
          teacher_body,
          mix=1.0,
          trust=float(getattr(c.cfg, "body_trust_region", 0.20)),
          max_norm=float(getattr(c.cfg, "body_max_norm", 0.0)),
        )
      packed = torch.cat((a_body, a_ee), dim=-1)
      act = full_teacher.clone()
      act[:, : c.body_dim] = exec_body
      if act.shape[-1] >= 17:
        act[:, 14:17] = 0.0
      elif act.shape[-1] == 14:
        act = torch.cat((act, torch.zeros(n, 3, device=c.device)), dim=-1)
      c.last_a = packed.detach()
      self._maybe_log(mode=mode, packed_a=packed)
      self._step += 1
      return act

    if c.hierarchical:
      if mode == "goal":
        p_sub = c.hier.p_goal_w
        c.hier.p_sub_w = p_sub.clone()
        c.hier.last_offset_w = c.hier.p_sub_w - c.hier.p_ee0_w
        high_a = c.hier.last_offset_w
      elif mode == "palm":
        palm_w = palm_pos_world(c.env, arm_left=c.arm_left)
        p_sub = palm_w
        c.hier.p_sub_w = p_sub.clone()
        c.hier.last_offset_w.zero_()
        high_a = torch.zeros(n, 3, device=c.device)
      else:
        high_a = c._high_level_action_legacy()
        p_sub = c.hier.apply_offset(
          high_a,
          scale_m=c.cfg.subgoal_offset_scale_m,
          rate_m=c.cfg.subgoal_rate_limit_m,
        )
      inject_subgoal_into_grasp(c.env, p_sub)
      c.env._homie_bptt_goal_w = c.hier.p_goal_w
      c.env._homie_bptt_sub_w = c.hier.p_sub_w
      obs_td = c.vec.get_observations()
      ee_delta = torch.zeros(n, 3, device=c.device)
      self._maybe_log(mode=mode, packed_a=high_a)
    else:
      high_a = c._high_level_action_legacy()
      ee_delta = high_a
      cmd_w = _residual_cmd_world(c.env, ee_delta, arm_left=c.arm_left)
      grasp = c.env.command_manager.get_term("grasp")
      c.env._homie_bptt_goal_w = grasp.target_pos_w
      c.env._homie_bptt_sub_w = cmd_w
      self._maybe_log(mode=mode, packed_a=high_a)

    with torch.inference_mode():
      full_a = c.policy(obs_td)
    from homierl_mjlab.rl.stage3_bptt.collector import meters_to_ee_raw

    ee_raw = meters_to_ee_raw(ee_delta, c.ee_scale)
    act = full_a.clone()
    if act.shape[-1] >= 17:
      act[:, 14:17] = ee_raw
    elif act.shape[-1] == 14:
      act = torch.cat((act, ee_raw), dim=-1)
    c.last_a = high_a.detach()
    self._step += 1
    return act


def _run_standard_mjlab_oracle(stage2: str) -> int:
  """Control-link test via standard mjlab.play (same as play_stage3_oracle_subgoal)."""
  import os

  os.environ["HOMIE_PLAY_ORACLE_SUBGOAL"] = "1"
  import homierl_mjlab  # noqa: F401

  argv = [
    "Mjlab-Homie-Flat-Unitree-G1-Phase3",
    "--num-envs",
    "1",
    "--viewer",
    "viser",
    "--checkpoint-file",
    stage2,
  ]
  print(
    "[bptt-play] ee-mode=goal → standard mjlab.play oracle "
    "(slow play IK, Viser owns grasp, no BPTT inject)\n"
    f"  argv: {argv}"
  )
  sys.argv = ["mjlab.scripts.play", *argv]
  from mjlab.scripts.play import main as play_main

  play_main()
  return 0


def main() -> int:
  args = _parse()
  need_actor = str(args.ee_mode) in ("actor", "attractor")
  payload: dict = {}
  bptt_path = Path(args.bptt_checkpoint) if args.bptt_checkpoint else None
  if bptt_path is not None and str(bptt_path):
    if not bptt_path.is_file():
      print(f"error: bptt checkpoint not found: {bptt_path}", file=sys.stderr)
      return 2
    payload = torch.load(bptt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "actor" not in payload:
      keys = sorted(payload.keys())[:12] if isinstance(payload, dict) else type(payload)
      print(
        "error: --bptt-checkpoint is not a Stage3 BPTT ckpt (need key 'actor').\n"
        "  Got keys/type: "
        f"{keys}\n"
        "  Hint: use a train_stage3_bptt run model_*.pt "
        "(not PPO phase3 HIM / model_state_dict).",
        file=sys.stderr,
      )
      return 2
  elif need_actor:
    print(
      "error: --bptt-checkpoint required for --ee-mode actor/attractor",
      file=sys.stderr,
    )
    return 2

  stage2 = args.stage2_checkpoint or str(payload.get("stage2_checkpoint") or "")
  if not stage2:
    print(
      "error: need --stage2-checkpoint (not stored in this bptt ckpt).\n"
      "  Stage3 actor is a_body+a_ee only; frozen phase-2 HIM is still the "
      "loco/teacher policy loaded into the env.",
      file=sys.stderr,
    )
    return 2
  if not Path(stage2).is_file():
    # Payload often stores a dated train path that was later moved/copied.
    candidates: list[str] = []
    alt = str(payload.get("stage2_checkpoint") or "")
    if alt and alt != stage2:
      candidates.append(alt)
    # Common local aliases used in this repo / launch.json.
    for rel in (
      "logs/rsl_rl/homie_g1/phase2/model_18600.pt",
      "logs/rsl_rl/homie_g1/phase2/model_18600(2).pt",
    ):
      candidates.append(str((_ROOT / rel) if not Path(rel).is_absolute() else Path(rel)))
      candidates.append(rel)
    resolved = ""
    for c in candidates:
      if c and Path(c).is_file():
        resolved = c
        break
    if resolved:
      print(
        f"[bptt-play] stage2 path missing ({stage2}); using {resolved}"
      )
      stage2 = resolved
    else:
      print(
        f"error: stage2 checkpoint not found: {stage2}\n"
        "  Pass --stage2-checkpoint <phase2/model_*.pt> "
        "(frozen HIM loco/teacher required even for a_body-only play).",
        file=sys.stderr,
      )
      return 2

  if str(args.ee_mode) == "goal":
    return _run_standard_mjlab_oracle(stage2)

  from homierl_mjlab.rl.stage3_bptt.cfg import (
    ACTION_SEMANTICS_BODY_EE,
    Stage3BpttCfg,
  )
  from homierl_mjlab.rl.stage3_bptt.collector import Stage3SimCollector
  from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor
  from mjlab.viewer import ViserPlayViewer
  from mjlab.viewer.base import VerbosityLevel

  saved_cfg = payload.get("cfg") or {}
  control_mode = (
    args.control_mode
    or payload.get("control_mode")
    or saved_cfg.get("control_mode")
    or "body_ee_palm_delta"
  )
  ckpt_sem = payload.get("action_semantics") or saved_cfg.get("action_semantics")
  if need_actor and control_mode == "body_ee_palm_delta":
    if ckpt_sem is None:
      print(
        "error: refusing to load Stage3 ckpt without action_semantics into "
        f"{ACTION_SEMANTICS_BODY_EE}. Old layouts are incompatible — "
        "retrain with loco-obs + raw a_body (v3).",
        file=sys.stderr,
      )
      return 2
    if ckpt_sem != ACTION_SEMANTICS_BODY_EE:
      print(
        f"error: action_semantics mismatch ckpt={ckpt_sem!r} "
        f"expected={ACTION_SEMANTICS_BODY_EE!r}\n"
        "  Prior tanh / reach-only actors need a fresh Stage3 train.",
        file=sys.stderr,
      )
      return 2

  cur = args.curriculum.upper()
  if cur not in ("A", "B"):
    cur = "A"

  cfg = Stage3BpttCfg(
    device=args.device,
    stage2_checkpoint=stage2,
    curriculum=cur,
    curriculum_auto=False,
    body_exec_teacher=False,
    control_mode=str(control_mode),
    arm_side=str(saved_cfg.get("arm_side", "right")),
    sdf_query_count=int(saved_cfg.get("sdf_query_count", 64)),
    subgoal_offset_scale_m=float(
      saved_cfg.get("subgoal_offset_scale_m", 0.55)
    ),
    subgoal_rate_limit_m=float(saved_cfg.get("subgoal_rate_limit_m", 0.04)),
    action_scale_m=float(saved_cfg.get("action_scale_m", 0.02)),
    ee_delta_scale_m=float(saved_cfg.get("ee_delta_scale_m", 0.03)),
    body_dim=int(saved_cfg.get("body_dim", 14)),
    ee_dim=int(saved_cfg.get("ee_dim", 3)),
    body_trust_region=float(saved_cfg.get("body_trust_region", 0.20)),
    body_max_norm=float(saved_cfg.get("body_max_norm", 0.0)),
    body_exec_mix=float(saved_cfg.get("body_exec_mix", 0.0)),
    teacher_action_norm=str(saved_cfg.get("teacher_action_norm", "none")),
    teacher_action_clip=float(saved_cfg.get("teacher_action_clip", 100.0)),
    body_action_clip=float(saved_cfg.get("body_action_clip", 100.0)),
  )
  cfg.action_semantics = cfg.resolve_action_semantics()
  if "actor_hidden" in saved_cfg:
    cfg.actor_hidden = tuple(saved_cfg["actor_hidden"])

  loco_src = str(args.loco_source)
  body_only = bool(args.body_only)
  zero_ee = bool(args.zero_ee_delta) or body_only or loco_src != "actor"
  subgoal = str(args.subgoal)
  if subgoal == "auto":
    if body_only:
      subgoal = "palm"
    elif zero_ee:
      subgoal = "goal"
    else:
      subgoal = "palm"
  # body-only always locks arms; body+ik (zero-ee + goal) uses oracle UX
  # (Arm chase checkbox after waist target).
  arm_chase_gate = (not body_only) and (subgoal == "goal" or not zero_ee)
  print(
    f"[bptt-play] control_mode={cfg.control_mode} semantics={cfg.action_semantics} "
    f"curriculum={cur} ee_mode={args.ee_mode} loco_source={loco_src} "
    f"body_only={body_only} zero_ee={zero_ee} subgoal={subgoal}"
  )
  if str(args.ee_mode) == "palm":
    print("[bptt-play] actor BYPASSED — p_sub := palm (stand / loco only)")
  elif need_actor and loco_src == "actor":
    print(
      "[bptt-play] loco from Stage3 a_body — if this flies, body head is not "
      "ready (train still in body/teacher-exec phase)"
    )
  elif need_actor and loco_src == "teacher":
    print("[bptt-play] loco from frozen phase-2 teacher (Stage3 a_body ignored)")
  print(f"[bptt-play] stage2={stage2}")
  print(f"[bptt-play] bptt={bptt_path if bptt_path else '(none)'}")
  print(
    "[bptt-play] spheres: orange=p_goal from Viser body xyz; "
    "cyan=p_sub (palm+delta or pinned). Prefer y<0 for right-arm ckpt."
  )

  collector = Stage3SimCollector(
    cfg,
    stage2_checkpoint=stage2,
    num_envs=int(args.num_envs),
    ee_mode="attractor",
    actor=None,
    play=True,
  )
  collector.ee_mode = str(args.ee_mode)
  if need_actor and loco_src == "actor":
    loco_dim = int(
      payload.get("loco_obs_dim", getattr(collector, "loco_obs_dim", 0)) or 0
    )
    if loco_dim <= 0:
      loco_dim = int(getattr(collector, "loco_obs_dim", 0) or 0)
    if int(collector.obs_dim) != int(
      payload.get("obs_dim", collector.obs_dim) or collector.obs_dim
    ):
      print(
        f"[bptt-play] warn: ckpt obs_dim={payload.get('obs_dim')} "
        f"vs collector={collector.obs_dim} (loco={loco_dim})"
      )
    actor = Stage3Actor(
      collector.obs_dim,
      cfg.actor_hidden,
      body_dim=cfg.body_dim,
      ee_dim=cfg.ee_dim,
      ee_delta_scale_m=cfg.ee_delta_scale_m,
      action_scale_m=collector.actor_action_scale,
      control_mode=cfg.control_mode,
      near_zero_body_init=False,
      loco_obs_dim=loco_dim,
      body_action_clip=float(getattr(cfg, "body_action_clip", 100.0)),
    ).to(collector.device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    for p_ in actor.parameters():
      p_.requires_grad_(False)
    collector.actor = actor
    print(
      f"[bptt-play] loaded Stage3 actor out_dim={getattr(actor, 'out_dim', '?')} "
      f"obs_dim={collector.obs_dim}"
    )
  else:
    collector.actor = None
    if need_actor:
      print(f"[bptt-play] Stage3 actor not loaded (loco_source={loco_src})")
    else:
      print("[bptt-play] Stage3 actor not loaded (control-link test)")

  collector.env._homie_play_arm_chase = False
  ee = collector.env.action_manager.get_term("ee_residual")
  if hasattr(ee, "_require_play_arm_chase"):
    ee._require_play_arm_chase = bool(arm_chase_gate)
  if (collector.hierarchical or collector.body_ee) and hasattr(ee, "_use_policy_delta"):
    ee._use_policy_delta = False
  if body_only or (zero_ee and subgoal == "palm"):
    print(
      "[bptt-play] a_ee:=0, p_sub:=palm, arm chase OFF "
      "(verify Stage3 a_body / lower-limb stability)"
    )
  elif zero_ee and subgoal == "goal":
    print(
      "[bptt-play] a_ee:=0, p_sub:=p_goal — Stage3 a_body + DLS "
      "(compare to oracle loco+IK). Arm chase OFF until Viser: "
      "Enable waist target → set XYZ → Arm chase ON"
    )
  else:
    print(
      "[bptt-play] Arm IK off until Viser: Enable waist target → set XYZ → "
      "Arm chase (slow) ON"
    )

  _install_dual_target_debug_vis(
    collector.env, show_sub=collector.hierarchical or collector.body_ee
  )
  if arm_chase_gate:
    collector._force_reach_flags()
  st = _goal_state(collector)
  st.ensure_all(collector.env, arm_left=collector.arm_left)
  collector.env._homie_bptt_goal_w = st.p_goal_w
  collector.env._homie_bptt_sub_w = st.p_sub_w

  play_env = _PlayEnvAdapter(collector)
  policy = Stage3BpttPlayPolicy(
    collector,
    log_every=int(args.log_every),
    body_only=body_only,
    zero_ee=zero_ee,
    subgoal=subgoal,
    loco_source=loco_src,
  )
  viewer = ViserPlayViewer(
    play_env, policy, verbosity=VerbosityLevel.INFO
  )
  print("[bptt-play] open the Viser URL printed by mjlab; Reset re-samples p_goal")
  viewer.run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
