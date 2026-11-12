"""Real-sim collector: Stage-3 body+EE (or legacy hierarchical / residual)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from homierl_mjlab.mdp.base_frame import world_to_base
from homierl_mjlab.mdp.sdf_observations import arm_clearance_field, sdf_query_field
from homierl_mjlab.rl.him_runner import HomieOnPolicyRunner
from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
from homierl_mjlab.rl.stage3_bptt.env_bridge import (
  attractor_delta,
  goal_geom_from_env,
  pack_robot_state,
  palm_pos_base,
)
from homierl_mjlab.rl.stage3_bptt.geom_clearance import (
  _present_from_geom,
  palm_probe_clearance,
  sdf_query_from_geom,
)
from homierl_mjlab.rl.stage3_bptt.hierarchical import (
  HierarchicalSubgoalState,
  PalmDeltaSubgoalState,
  attractor_offset_w,
  attractor_palm_delta_b,
  body_ee_obs_dim,
  build_body_ee_obs,
  build_hierarchical_obs,
  hierarchical_obs_dim,
  inject_subgoal_into_grasp,
)
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import (
  mean_terms,
  soft_stability_reward,
  stage3_reward,
  tilt_angle_from_root_quat,
)


def _configure_grasp_reach(env_cfg) -> None:
  grasp = env_cfg.commands.get("grasp")
  if grasp is not None:
    grasp.enable_arm_reach = True
    grasp.init_reach_ratio = 1.0
    grasp.mode_probs_start = (0.0, 0.0, 1.0)
    grasp.mode_probs = (0.0, 0.0, 1.0)
    grasp.init_bend_ratio = 1.0


def _configure_curriculum(env_cfg, cfg: Stage3BpttCfg) -> None:
  """A / auto-start: no crate. B: crate mix. Auto later raises presence live."""
  _configure_grasp_reach(env_cfg)
  cont = env_cfg.commands.get("container")
  if cont is None:
    return
  auto = bool(getattr(cfg, "curriculum_auto", False))
  cur = str(cfg.curriculum).upper()
  if auto or cur == "A":
    cont.presence_prob = 0.0
    cont.debug_vis = False
    # Still wire crate flags so live presence bumps work when auto advances.
    if auto:
      cont.presence_requires_reach = True
      cont.presence_requires_need_waist = True
      cont.presence_also_on_low_z = True
      cont.relocate_grasp_when_present = bool(cfg.relocate_grasp_when_present)
    return
  cont.presence_prob = float(cfg.curriculum_b_presence_prob)
  cont.presence_requires_reach = True
  cont.presence_requires_need_waist = True
  cont.presence_also_on_low_z = True
  cont.relocate_grasp_when_present = bool(cfg.relocate_grasp_when_present)
  cont.debug_vis = False
  actions = getattr(env_cfg, "actions", None)
  ee = actions.get("ee_residual") if isinstance(actions, dict) else None
  if ee is not None:
    if hasattr(ee, "avoid_d_safe"):
      ee.avoid_d_safe = float(cfg.d_safe)
    if hasattr(ee, "avoid_lift_margin"):
      ee.avoid_lift_margin = float(cfg.avoid_lift_margin)
    if hasattr(ee, "avoid_repulsion_gain"):
      ee.avoid_repulsion_gain = float(cfg.avoid_repulsion_gain)
    if hasattr(ee, "avoid_xy_slow"):
      ee.avoid_xy_slow = float(cfg.avoid_xy_slow)


def _active_arm_clearance(
  clr6: torch.Tensor, *, arm_left: bool
) -> torch.Tensor:
  return clr6[:, :3] if arm_left else clr6[:, 3:6]


def meters_to_ee_raw(
  delta_m: torch.Tensor, scale: tuple[float, float, float]
) -> torch.Tensor:
  """Invert ``delta = scale * tanh(raw)`` for EeResidualIkAction."""
  s = torch.tensor(scale, device=delta_m.device, dtype=delta_m.dtype)
  x = torch.clamp(delta_m / s.clamp_min(1e-6), -0.999, 0.999)
  return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def normalize_teacher_actions(
  actions: torch.Tensor,
  *,
  mode: str = "tanh",
  clip: float = 1.0,
) -> torch.Tensor:
  """Map HIM teacher means into Stage3 body space (≈ tanh / [-1, 1]).

  HIM ``act_inference`` only clamps to ±100 for numerics; Stage3 ``a_body`` is
  ``tanh``-normalized. BC / curriculum MSE / trust-region all require the same
  scale — hard clip saturates, soft ``tanh`` preserves ranking.
  """
  m = str(mode).lower().strip()
  if m in ("tanh", "soft", "normalize"):
    return torch.tanh(actions)
  c = float(clip)
  return actions.clamp(-c, c)


def safe_exec_body(
  a_body: torch.Tensor,
  teacher_body: torch.Tensor,
  *,
  mix: float,
  trust: float,
  max_norm: float = 0.0,
) -> torch.Tensor:
  """Blend actor body into **normalized** teacher, then trust-region.

  Both ``a_body`` (tanh) and ``teacher_body`` must already be in the same
  [-1, 1]-ish space (see ``normalize_teacher_actions``). Then::

    exec = teacher + clip(blend(actor, teacher) - teacher, ±trust)

  ``mix=0`` → pure teacher. ``mix=1`` + ``trust>0`` → actor stays near teacher
  (prevents takeoff from an unnormalized / untrained a_body).
  """
  mix = float(max(0.0, min(1.0, mix)))
  if mix <= 1e-8:
    return teacher_body
  a = a_body
  if max_norm > 1e-8:
    n = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(1e-6)
    a = a * (float(max_norm) / n).clamp(max=1.0)
  blended = (1.0 - mix) * teacher_body + mix * a
  trust = float(trust)
  if trust <= 1e-8:
    return blended
  return teacher_body + (blended - teacher_body).clamp(-trust, trust)


def _scalarize_log_val(v) -> float:
  if isinstance(v, torch.Tensor):
    return float(v.detach().float().mean().cpu())
  return float(v)


def _aggregate_ep_logs(ep_logs: list[dict]) -> dict[str, float]:
  if not ep_logs:
    return {}
  keys = {k for ep in ep_logs for k in ep}
  out: dict[str, float] = {}
  for key in sorted(keys):
    vals = [_scalarize_log_val(ep[key]) for ep in ep_logs if key in ep]
    if vals:
      out[f"ppo/{key}"] = sum(vals) / len(vals)
  return out


class Stage3SimCollector:
  """Phase-3 env + frozen phase-2 HIM teacher + Stage-3 actor/attractor."""

  def __init__(
    self,
    cfg: Stage3BpttCfg,
    *,
    stage2_checkpoint: str,
    num_envs: int = 64,
    ee_mode: Literal["attractor", "actor"] = "attractor",
    actor: Stage3Actor | None = None,
    play: bool = False,
  ):
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.ee_mode = ee_mode
    self.arm_left = cfg.arm_side == "left"
    self.actor = actor
    self.play = bool(play)

    env_cfg = load_env_cfg(
      "Mjlab-Homie-Flat-Unitree-G1-Phase3", play=self.play
    )
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(cfg.seed)
    _configure_curriculum(env_cfg, cfg)
    self.control_mode = str(
      getattr(cfg, "control_mode", "body_ee_palm_delta")
    )
    self.body_ee = self.control_mode == "body_ee_palm_delta"
    self.hierarchical = self.control_mode == "hierarchical_subgoal"
    # Progressive curriculum: start with arm locked unless auto disabled.
    auto = bool(getattr(cfg, "curriculum_auto", False)) and self.body_ee
    self.arm_unlocked = not auto
    self._presence_prob = (
      0.0
      if auto or str(cfg.curriculum).upper() == "A"
      else float(cfg.curriculum_b_presence_prob)
    )
    if self.play:
      ee = env_cfg.actions.get("ee_residual") if hasattr(env_cfg, "actions") else None
      if ee is not None:
        ee.require_play_arm_chase = True
        if self.hierarchical or self.body_ee:
          # Subgoal injected into grasp; residual stays 0 (no double EE).
          ee.use_policy_delta = False
          ee.avoid_container = False

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(self.device))
    self.vec = RslRlVecEnvWrapper(raw_env)
    self.env = self.vec.unwrapped

    rl_cfg = load_rl_cfg("Mjlab-Homie-Flat-Unitree-G1-Phase3")
    train_cfg = asdict(rl_cfg)

    self.runner = HomieOnPolicyRunner(
      self.vec, train_cfg, log_dir=None, device=str(self.device)
    )
    if not stage2_checkpoint:
      raise ValueError("stage2_checkpoint is required for frozen loco / teacher")
    path = Path(stage2_checkpoint)
    if not path.is_file():
      raise FileNotFoundError(path)
    self.runner.load(str(path), strict=False)
    if not self.play:
      self.runner._maybe_setup_teacher()
    self.runner.alg.actor_critic.eval()
    for p in self.runner.alg.actor_critic.parameters():
      p.requires_grad_(False)
    self.policy = self.runner.get_inference_policy(device=str(self.device))

    ee_cfg = self.env.action_manager.get_term("ee_residual").cfg
    self.ee_scale = tuple(float(x) for x in ee_cfg.ee_delta_scale)

    robot = self.env.scene["robot"]
    self.n_joints = int(robot.data.joint_pos.shape[-1])
    self._state0 = pack_robot_state(self.env, arm_left=self.arm_left)
    self.state_dim = int(self._state0.shape[-1])
    self.control_dim = self.n_joints
    self.goal_dim = 3
    self.geom_dim = 10
    self.body_dim = int(getattr(cfg, "body_dim", 14))
    self.ee_dim = int(getattr(cfg, "ee_dim", 3))
    self.actor_out_dim = (
      self.body_dim + self.ee_dim if self.body_ee else 3
    )

    if self.body_ee:
      self.obs_dim = body_ee_obs_dim(
        cfg.sdf_query_count, action_dim=self.actor_out_dim
      )
      self.actor_action_scale = float(cfg.ee_delta_scale_m)
    elif self.hierarchical:
      self.obs_dim = hierarchical_obs_dim(cfg.sdf_query_count)
      self.actor_action_scale = float(cfg.subgoal_offset_scale_m)
    else:
      self.obs_dim = 3 + 3 + 3 + cfg.sdf_query_count + 6
      self.actor_action_scale = float(cfg.action_scale_m)
    self.feat_dim = self.state_dim + self.goal_dim + self.geom_dim

    self.last_a = torch.zeros(
      self.env.num_envs, self.actor_out_dim, device=self.device
    )
    self._env_ep_id = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.long
    )
    self._ep_len = torch.zeros(self.env.num_envs, device=self.device)
    self._last_mean_ep_len = 0.0
    self.hier = HierarchicalSubgoalState(self.env.num_envs, self.device)
    self.palm_state = PalmDeltaSubgoalState(self.env.num_envs, self.device)
    if self.body_ee:
      self.palm_state.ensure_all(self.env, arm_left=self.arm_left)
    else:
      self.hier.ensure_all(self.env, arm_left=self.arm_left)

  def set_arm_unlocked(self, unlocked: bool) -> None:
    self.arm_unlocked = bool(unlocked)

  def set_presence_prob(self, prob: float) -> None:
    """Live-update container presence (progressive curriculum)."""
    p = float(max(0.0, min(1.0, prob)))
    self._presence_prob = p
    try:
      cont = self.env.command_manager.get_term("container")
      cont.cfg.presence_prob = p
      if hasattr(cont, "presence_prob"):
        # Some terms mirror cfg onto an attribute used at resample.
        cont.presence_prob = p
    except Exception:  # noqa: BLE001
      pass

  def max_episode_length_steps(self) -> float:
    """Approx policy steps per episode (for body-phase ep_len gate)."""
    try:
      cfg = self.env.cfg
      ep_s = float(getattr(cfg, "episode_length_s", 0.0) or 0.0)
      dt = float(self.env.step_dt)
      if ep_s > 0.0 and dt > 0.0:
        return max(ep_s / dt, 1.0)
    except Exception:  # noqa: BLE001
      pass
    return 500.0

  def _force_reach_flags(self) -> None:
    grasp = self.env.command_manager.get_term("grasp")
    grasp.is_target[:] = 1.0
    grasp.is_waist[:] = 1.0
    grasp.waist_on = True

  def _sdf_clr(self) -> tuple[torch.Tensor, torch.Tensor]:
    try:
      sdf = sdf_query_field(
        self.env,
        query_count=self.cfg.sdf_query_count,
        d_max=self.cfg.d_max,
        include_valid=False,
      )
    except Exception:  # noqa: BLE001
      sdf = torch.zeros(
        self.env.num_envs, self.cfg.sdf_query_count, device=self.device
      )
    try:
      clr = arm_clearance_field(self.env, d_max=self.cfg.d_max)
    except Exception:  # noqa: BLE001
      clr = torch.ones(self.env.num_envs, 6, device=self.device) * self.cfg.d_max
    return sdf, clr

  def _build_actor_obs(self) -> torch.Tensor:
    sdf, clr = self._sdf_clr()
    if self.body_ee:
      return build_body_ee_obs(
        self.env,
        self.palm_state,
        arm_left=self.arm_left,
        sdf=sdf,
        clearance=clr,
        last_a=self.last_a,
      )
    if self.hierarchical:
      return build_hierarchical_obs(
        self.env,
        self.hier,
        arm_left=self.arm_left,
        sdf=sdf,
        clearance=clr,
      )
    palm = palm_pos_base(self.env, arm_left=self.arm_left)
    goal, _ = goal_geom_from_env(self.env)
    return torch.cat((palm, goal, self.last_a[..., :3], sdf, clr), dim=-1)

  def _actor_body_ee(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(a_body, a_ee_delta_m)`` from actor or attractor+teacher."""
    n = self.env.num_envs
    if self.ee_mode == "actor" and self.actor is not None:
      obs = self._build_actor_obs()
      with torch.no_grad():
        out = self.actor(obs)
      if isinstance(out, tuple):
        return out[0], out[1]
      # Legacy 3-D actor misconfigured — treat as EE only.
      return torch.zeros(n, self.body_dim, device=self.device), out

    # Attractor: EE toward true goal; body from frozen teacher after inject.
    a_ee = attractor_palm_delta_b(
      self.env,
      self.palm_state,
      arm_left=self.arm_left,
      scale_m=min(self.cfg.ee_delta_scale_m, 0.02),
    )
    # Placeholder body; overwritten after inject + teacher query in collect.
    a_body = torch.zeros(n, self.body_dim, device=self.device)
    return a_body, a_ee

  def _high_level_action_legacy(self) -> torch.Tensor:
    """3-D hierarchical / residual path."""
    if self.ee_mode == "actor" and self.actor is not None:
      obs = self._build_actor_obs()
      with torch.no_grad():
        out = self.actor(obs)
      return out if not isinstance(out, tuple) else out[1]
    if self.hierarchical:
      return attractor_offset_w(
        self.env,
        self.hier,
        arm_left=self.arm_left,
        scale_m=min(self.cfg.subgoal_offset_scale_m, 0.25),
        clip_m=self.cfg.subgoal_offset_scale_m,
      )
    delta = attractor_delta(
      self.env,
      arm_left=self.arm_left,
      scale=min(self.cfg.action_scale_m, 0.015),
      clip=self.cfg.action_scale_m,
    )
    if str(self.cfg.curriculum).upper() != "A":
      try:
        from homierl_mjlab.mdp.palm import (
          LEFT_EE_LINK,
          RIGHT_EE_LINK,
          link_point_pos_w,
          palm_offset_tensor,
        )
        from homierl_mjlab.mdp.sdf_observations import container_ik_avoidance_bias

        robot = self.env.scene["robot"]
        ee_name = LEFT_EE_LINK if self.arm_left else RIGHT_EE_LINK
        ids, _ = robot.find_bodies(ee_name, preserve_order=True)
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        off = palm_offset_tensor(
          self.device, robot.data.joint_pos.dtype, left=self.arm_left
        )
        ee_w = link_point_pos_w(robot, int(ids[0]), off, env_ids)
        grasp = self.env.command_manager.get_term("grasp")
        bias = container_ik_avoidance_bias(
          self.env,
          env_ids,
          ee_w,
          grasp.target_pos_w,
          d_safe=self.cfg.d_safe,
          lift_margin=float(self.cfg.avoid_lift_margin),
          xy_slow=float(self.cfg.avoid_xy_slow),
          repulsion_gain=float(self.cfg.avoid_repulsion_gain),
          d_max=self.cfg.d_max,
        )
        delta = delta + bias
      except Exception:  # noqa: BLE001
        pass
    return delta

  def _true_goal_base(self) -> torch.Tensor:
    env_ids = torch.arange(self.env.num_envs, device=self.device)
    if self.body_ee:
      return world_to_base(
        self.env.scene["robot"], self.palm_state.p_goal_w, env_ids
      )
    if self.hierarchical:
      return world_to_base(self.env.scene["robot"], self.hier.p_goal_w, env_ids)
    goal, _ = goal_geom_from_env(self.env)
    return goal

  def _applied_joint_targets(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    u = robot.data.default_joint_pos.clone()
    jp = self.env.action_manager.get_term("joint_pos")
    tids = jp.target_ids
    u[:, tids] = jp._processed_actions - robot.data.encoder_bias[:, tids]
    ee = self.env.action_manager.get_term("ee_residual")
    if self.arm_left:
      u[:, ee._left_ids] = ee._left_cmd
    else:
      u[:, ee._right_ids] = ee._right_cmd
    return u

  def collect(self, replay: EpisodeReplay, num_steps: int) -> dict[str, float]:
    self._force_reach_flags()
    if self.body_ee:
      self.palm_state.ensure_all(self.env, arm_left=self.arm_left)
    else:
      self.hier.ensure_all(self.env, arm_left=self.arm_left)
    obs_td = self.vec.get_observations()
    n = self.env.num_envs
    coll_env_sum = 0.0
    present_sum = 0.0
    rew_sum = 0.0
    ppo_step_rew_sum = 0.0
    reach_sum = 0.0
    clear_reach_sum = 0.0
    dist_sum = 0.0
    teacher_mse_sum = 0.0
    body_norm_sum = 0.0
    ee_norm_sum = 0.0
    term_sums: dict[str, float] = {}
    ep_logs: list[dict] = []
    finished_ep_lens: list[float] = []

    for _step_i in range(int(num_steps)):
      state = pack_robot_state(self.env, arm_left=self.arm_left)
      _, geom = goal_geom_from_env(self.env)
      goal = self._true_goal_base()

      teacher_body = torch.zeros(n, self.body_dim, device=self.device)
      packed_a = torch.zeros(n, self.actor_out_dim, device=self.device)

      if self.body_ee:
        a_body, a_ee = self._actor_body_ee()
        if not self.arm_unlocked:
          # Body-only phase: hold grasp on current palm (no EE chase).
          a_ee = torch.zeros(n, 3, device=self.device)
          from homierl_mjlab.rl.stage3_bptt.hierarchical import palm_pos_world

          p_sub = palm_pos_world(self.env, arm_left=self.arm_left)
          self.palm_state.p_sub_w = p_sub.clone()
          self.palm_state.last_delta_b.zero_()
        else:
          p_sub = self.palm_state.apply_palm_delta_base(
            self.env,
            a_ee,
            arm_left=self.arm_left,
            rate_m=self.cfg.subgoal_rate_limit_m,
          )
        inject_subgoal_into_grasp(self.env, p_sub)
        # Teacher sees post-inject grasp so body tracks p_sub-conditioned loco.
        obs_td = self.vec.get_observations()
        with torch.inference_mode():
          full_teacher = self.policy(obs_td)
        # Normalize HIM (±100 numeric) → Stage3 tanh space before BC / exec.
        full_teacher = normalize_teacher_actions(
          full_teacher,
          mode=str(getattr(self.cfg, "teacher_action_norm", "tanh")),
          clip=float(getattr(self.cfg, "teacher_action_clip", 1.0)),
        )
        teacher_body = full_teacher[..., : self.body_dim].detach()
        if self.ee_mode != "actor" or self.actor is None:
          a_body = teacher_body.clone()
        packed_a = torch.cat((a_body, a_ee), dim=-1)
        # Sim exec: never apply raw a_body. Normalized teacher (+ trust-region)
        # keeps feet on the ground; packed_a still stores actor for BC / critic.
        pure_teacher = (not self.arm_unlocked) and bool(
          getattr(self.cfg, "body_exec_teacher", True)
        )
        mix = (
          float(getattr(self.cfg, "body_exec_mix", 0.0))
          if pure_teacher
          else 1.0
        )
        exec_body = safe_exec_body(
          a_body,
          teacher_body,
          mix=mix,
          trust=float(getattr(self.cfg, "body_trust_region", 0.20)),
          max_norm=float(getattr(self.cfg, "body_max_norm", 2.5)),
        )
        act = full_teacher.clone()
        act[:, : self.body_dim] = exec_body
        # Env policy action is 14-D (IK residual off). Only pad if the env
        # still exposes a 3-D EE action slot (legacy use_policy_delta=True).
        env_act = int(self.env.action_manager.total_action_dim)
        if env_act >= 17 and act.shape[-1] >= 17:
          act[:, 14:17] = 0.0
        elif env_act >= 17 and act.shape[-1] == 14:
          act = torch.cat(
            (act, torch.zeros(n, 3, device=self.device)), dim=-1
          )
        elif act.shape[-1] > env_act:
          act = act[:, :env_act]
        smooth_a = packed_a
      else:
        high_a = self._high_level_action_legacy()
        if self.hierarchical:
          p_sub = self.hier.apply_offset(
            high_a,
            scale_m=self.cfg.subgoal_offset_scale_m,
            rate_m=self.cfg.subgoal_rate_limit_m,
          )
          inject_subgoal_into_grasp(self.env, p_sub)
          obs_td = self.vec.get_observations()
          ee_delta = torch.zeros(n, 3, device=self.device)
        else:
          ee_delta = high_a
        with torch.inference_mode():
          full_a = self.policy(obs_td)
        full_a = normalize_teacher_actions(
          full_a,
          mode=str(getattr(self.cfg, "teacher_action_norm", "tanh")),
          clip=float(getattr(self.cfg, "teacher_action_clip", 1.0)),
        )
        teacher_body = full_a[..., : self.body_dim].detach()
        ee_raw = meters_to_ee_raw(ee_delta, self.ee_scale)
        act = full_a.clone()
        env_act = int(self.env.action_manager.total_action_dim)
        if env_act >= 17 and act.shape[-1] >= 17:
          act[:, 14:17] = ee_raw
        elif env_act >= 17 and act.shape[-1] == 14:
          act = torch.cat((act, ee_raw), dim=-1)
        elif act.shape[-1] > env_act:
          act = act[:, :env_act]
        packed_a = high_a
        smooth_a = high_a

      obs_td, ppo_rew, dones, extras = self.vec.step(act)
      self._force_reach_flags()
      control = self._applied_joint_targets()
      next_state = pack_robot_state(self.env, arm_left=self.arm_left)

      palm = palm_pos_base(self.env, arm_left=self.arm_left)
      try:
        clr6 = arm_clearance_field(self.env, d_max=self.cfg.d_max)
      except Exception:  # noqa: BLE001
        clr6 = torch.ones(n, 6, device=self.device) * self.cfg.d_max
      clr = _active_arm_clearance(clr6, arm_left=self.arm_left)
      try:
        present = self.env.command_manager.get_term("container").present
      except Exception:  # noqa: BLE001
        present = torch.zeros(n, device=self.device)
      stab = None
      if float(self.cfg.w_stability) != 0.0:
        quat = state[..., -16:-12]
        tilt = tilt_angle_from_root_quat(quat)
        stab = soft_stability_reward(
          tilt,
          limit_rad=self.cfg.stability_limit_rad,
          soft_rad=self.cfg.stability_soft_rad,
        )
      # Smooth uses EE slice when body_ee so loco BC noise does not dominate.
      smooth_now = smooth_a[..., -3:] if self.body_ee else smooth_a
      smooth_prev = self.last_a[..., -3:] if self.body_ee else self.last_a
      # Body phase: drop Stage3 reach terms (arm locked); keep stability/safe.
      reach_on = bool(self.arm_unlocked) or not self.body_ee
      r_s3, terms = stage3_reward(
        palm_pos=palm,
        goal_pos=goal,
        clearance=clr,
        action=smooth_now,
        last_action=smooth_prev,
        sigma_p=self.cfg.sigma_p,
        sigma_far=self.cfg.sigma_far,
        d_safe=self.cfg.d_safe,
        d_max=self.cfg.d_max,
        reach_threshold=self.cfg.reach_threshold,
        w_target=self.cfg.w_target if reach_on else 0.0,
        w_approach=self.cfg.w_approach if reach_on else 0.0,
        w_safe=self.cfg.w_safe,
        w_margin=self.cfg.w_margin,
        w_smooth=self.cfg.w_smooth if reach_on else 0.0,
        w_soft_success=self.cfg.w_soft_success if reach_on else 0.0,
        w_stability=self.cfg.w_stability,
        stability=stab,
        present=present,
        return_terms=True,
      )
      # Phase-1/2 loco stack from env (tracking / orientation / height / …).
      ppo = ppo_rew.reshape(n).to(dtype=r_s3.dtype, device=r_s3.device)
      w_loco = float(getattr(self.cfg, "w_loco", 1.0))
      r = r_s3 + w_loco * ppo
      terms["r"] = r
      terms["ppo_loco"] = ppo
      terms["w_loco"] = w_loco * ppo
      for k, v in mean_terms(terms, prefix="real/").items():
        term_sums[k] = term_sums.get(k, 0.0) + v
      dist = torch.linalg.norm(palm - goal, dim=-1)
      reach_ok = (dist < float(self.cfg.reach_threshold)).float()
      clear_ok = reach_ok * ((clr.min(dim=-1).values >= 0.0) | (present < 0.5)).float()
      reach_sum += float(reach_ok.mean().item())
      clear_reach_sum += float(clear_ok.mean().item())
      dist_sum += float(dist.mean().item())
      if self.body_ee:
        body_slice = packed_a[..., : self.body_dim]
        ee_slice = packed_a[..., self.body_dim :]
        teacher_mse_sum += float(
          torch.mean((body_slice - teacher_body) ** 2).item()
        )
        body_norm_sum += float(body_slice.norm(dim=-1).mean().item())
        ee_norm_sum += float(ee_slice.norm(dim=-1).mean().item())

      dones_f = dones.float()
      truncated = torch.zeros_like(dones_f)
      if isinstance(extras, dict) and "time_outs" in extras:
        truncated = extras["time_outs"].float()
      terminated = (dones_f - truncated).clamp(min=0.0)
      if isinstance(extras, dict):
        log = extras.get("log") or extras.get("episode")
        if log:
          ep_logs.append(log)
      env_ids = torch.arange(n, device=self.device)
      replay.add_batch(
        state=state,
        control=control,
        next_state=next_state,
        reward=r,
        terminated=terminated,
        truncated=truncated,
        goal=goal,
        geom=geom,
        present=present,
        env_id=env_ids,
        episode_id=self._env_ep_id,
        actor_action=packed_a.detach(),
        teacher_body=teacher_body.detach(),
      )
      self.last_a = packed_a.detach()
      if self.body_ee:
        self.palm_state.last_body = packed_a[..., : self.body_dim].detach()
      self._ep_len += 1.0
      done_mask = dones.bool()
      if bool(done_mask.any()):
        done_ids = done_mask.nonzero(as_tuple=False).flatten()
        finished_ep_lens.extend(self._ep_len[done_mask].detach().cpu().tolist())
        self._ep_len[done_mask] = 0.0
        self._env_ep_id[done_mask] += 1
        self.last_a[done_mask] = 0.0
        if self.body_ee:
          self.palm_state.reset_envs(self.env, done_ids, arm_left=self.arm_left)
        else:
          self.hier.reset_envs(self.env, done_ids, arm_left=self.arm_left)
        if hasattr(self.policy, "reset"):
          self.policy.reset()
      rew_sum += float(r.mean().item())
      ppo_step_rew_sum += float(ppo_rew.mean().item())
      hit = (clr.min(dim=-1).values < 0.0) & (present > 0.5)
      coll_env_sum += float(hit.float().mean().item())
      present_sum += float((present > 0.5).float().mean().item())

    denom = max(num_steps, 1)
    out = {
      "collect_steps": float(num_steps),
      "collect_rew_mean": rew_sum / denom,
      "ppo_step_rew_mean": ppo_step_rew_sum / denom,
      "collect_neg_clearance_frac": coll_env_sum / denom,
      "collect_present_frac": present_sum / denom,
      "reach_success_frac": reach_sum / denom,
      "clear_reach_success_frac": clear_reach_sum / denom,
      "mean_goal_distance": dist_sum / denom,
      "replay_size": float(replay.size),
      "ppo_ep_log_count": float(len(ep_logs)),
      "control_mode_body_ee": 1.0 if self.body_ee else 0.0,
      "control_mode_hierarchical": 1.0 if self.hierarchical else 0.0,
      "arm_unlocked": 1.0 if self.arm_unlocked else 0.0,
      "presence_prob": float(self._presence_prob),
      "w_loco": float(getattr(self.cfg, "w_loco", 1.0)),
      "body_exec_teacher": (
        1.0
        if (
          self.body_ee
          and (not self.arm_unlocked)
          and bool(getattr(self.cfg, "body_exec_teacher", True))
        )
        else 0.0
      ),
      "body_exec_mix": float(
        getattr(self.cfg, "body_exec_mix", 0.0)
        if (
          self.body_ee
          and (not self.arm_unlocked)
          and bool(getattr(self.cfg, "body_exec_teacher", True))
        )
        else (1.0 if self.body_ee else 0.0)
      ),
      "body_trust_region": float(
        getattr(self.cfg, "body_trust_region", 0.20) if self.body_ee else 0.0
      ),
    }
    if self.body_ee:
      out["teacher_mse_collect"] = teacher_mse_sum / denom
      out["actor_body_norm"] = body_norm_sum / denom
      out["actor_ee_norm"] = ee_norm_sum / denom
    for k, v in term_sums.items():
      out[k] = v / denom
    if finished_ep_lens:
      self._last_mean_ep_len = float(sum(finished_ep_lens) / len(finished_ep_lens))
      out["mean_episode_length"] = self._last_mean_ep_len
      out["episode_length_count"] = float(len(finished_ep_lens))
    else:
      out["mean_episode_length"] = float(self._last_mean_ep_len)
      out["episode_length_count"] = 0.0
    out.update(_aggregate_ep_logs(ep_logs))
    return out

  def make_fns(self):
    """Callables for Stage3BpttTrainer imagination.

    Returns ``(control_fn, obs_fn, feat_fn, geom_fn, palm_step_fn)``.
    """
    arm_left = self.arm_left
    n_j = self.n_joints
    ee = self.env.action_manager.get_term("ee_residual")
    arm_ids = ee._left_ids if arm_left else ee._right_ids
    jp = self.env.action_manager.get_term("joint_pos")
    loco_ids = jp.target_ids
    body_ee = self.body_ee
    hierarchical = self.hierarchical
    rate_m = float(self.cfg.subgoal_rate_limit_m)
    max_dq = float(self.cfg.max_joint_delta)
    body_dim = self.body_dim
    body_scale = float(getattr(self.cfg, "body_action_scale", 0.25))
    cart_to_joint = 2.0
    default_q = self.env.scene["robot"].data.default_joint_pos[0].detach().clone()

    def palm_step_fn(state, action, goal, geom):
      """Differentiable palm move: palm+delta (body_ee) / hier / residual."""
      del goal, geom
      palm = state[..., -3:]
      if body_ee:
        delta = action[..., -3:] if action.shape[-1] > 3 else action
        step = torch.clamp(delta, -rate_m, rate_m)
        return palm + step
      if hierarchical:
        target = action[..., -3:] if action.shape[-1] > 3 else action
        err = target - palm
        n = torch.linalg.norm(err, dim=-1, keepdim=True).clamp_min(1e-9)
        step = err * (rate_m / n).clamp(max=1.0)
        return palm + step
      step = torch.clamp(
        action[..., -3:] if action.shape[-1] > 3 else action,
        -rate_m,
        rate_m,
      )
      return palm + step

    def control_fn(state, action, goal, geom):
      del goal, geom
      q = state[..., :n_j]
      palm = state[..., -3:]
      u = q.clone()
      if body_ee:
        a_body = action[..., :body_dim]
        a_ee = action[..., body_dim : body_dim + 3]
        # Homie decode on loco joints: q = q_default + scale * a_body.
        u = u.clone()
        def_q = default_q.to(device=u.device, dtype=u.dtype).unsqueeze(0).expand_as(u)
        u[:, loco_ids] = def_q[:, loco_ids] + body_scale * a_body
        err = a_ee
        k = min(3, int(arm_ids.numel()))
        if k > 0:
          dq = torch.clamp(err[:, :k] * cart_to_joint, -max_dq, max_dq)
          idx = arm_ids[:k]
          u[:, idx] = u[:, idx] + dq
        return u
      delta_p = action[..., -3:] if action.shape[-1] > 3 else action
      if hierarchical:
        err = delta_p - palm
      else:
        err = delta_p
      k = min(3, int(arm_ids.numel()))
      if k > 0:
        dq = torch.clamp(err[:, :k] * cart_to_joint, -max_dq, max_dq)
        u = u.clone()
        idx = arm_ids[:k]
        u[:, idx] = u[:, idx] + dq
      return u

    d_max = float(self.cfg.d_max)
    sdf_k = int(self.cfg.sdf_query_count)
    actor_out = self.actor_out_dim

    def obs_fn(state, goal, geom, last_a):
      palm = state[..., -3:]
      b = state.shape[0]
      if geom is not None and geom.shape[-1] >= 10:
        present = _present_from_geom(geom).reshape(b, 1)
        sdf = sdf_query_from_geom(geom, query_count=sdf_k, d_max=d_max) * present
        clr3 = palm_probe_clearance(palm, geom, d_max=d_max)
        clr3 = torch.where(present > 0.5, clr3, torch.ones_like(clr3) * d_max)
        pad = torch.ones(b, 3, device=state.device, dtype=state.dtype) * d_max
        clr = (
          torch.cat((clr3, pad), dim=-1)
          if arm_left
          else torch.cat((pad, clr3), dim=-1)
        )
      else:
        sdf = torch.zeros(b, sdf_k, device=state.device)
        clr = torch.ones(b, 6, device=state.device) * d_max
      if last_a.shape[-1] != actor_out:
        pad_a = torch.zeros(b, actor_out, device=state.device, dtype=state.dtype)
        n_copy = min(actor_out, last_a.shape[-1])
        pad_a[..., :n_copy] = last_a[..., :n_copy]
        last_a = pad_a
      if body_ee:
        # palm, goal, goal-palm, prev_sub-palm≈0, last_a, sdf, clr
        zero = torch.zeros_like(palm)
        return torch.cat(
          (palm, goal[..., :3], goal[..., :3] - palm, zero, last_a, sdf, clr),
          dim=-1,
        )
      if hierarchical:
        return torch.cat(
          (palm, goal[..., :3], goal[..., :3], palm, last_a[..., :3], sdf, clr),
          dim=-1,
        )
      return torch.cat((palm, goal[..., :3], last_a[..., :3], sdf, clr), dim=-1)

    def feat_fn(state, goal, geom):
      return torch.cat((state, goal, geom), dim=-1)

    def geom_fn(state, goal, geom):
      del goal
      palm = state[..., -3:]
      if geom is not None and geom.shape[-1] >= 10:
        clearance = palm_probe_clearance(palm, geom, d_max=d_max)
      else:
        clearance = torch.ones(state.shape[0], 3, device=state.device) * d_max
      return palm, clearance

    return control_fn, obs_fn, feat_fn, geom_fn, palm_step_fn
