"""Real-sim collector: frozen phase-2 loco + Stage-3 EE (attractor or actor)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from tensordict import TensorDict

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
  palm_probe_clearance,
  sdf_query_from_geom,
)
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import stage3_reward


def _configure_grasp_reach(env_cfg) -> None:
  grasp = env_cfg.commands.get("grasp")
  if grasp is not None:
    grasp.enable_arm_reach = True
    grasp.init_reach_ratio = 1.0
    grasp.mode_probs_start = (0.0, 0.0, 1.0)
    grasp.mode_probs = (0.0, 0.0, 1.0)
    grasp.init_bend_ratio = 1.0


def _configure_curriculum(env_cfg, cfg: Stage3BpttCfg) -> None:
  """A: no crate. B: milder crate mix + lift-over IK so data is not all collisions."""
  _configure_grasp_reach(env_cfg)
  cont = env_cfg.commands.get("container")
  if cont is None:
    return
  cur = str(cfg.curriculum).upper()
  if cur == "A":
    cont.presence_prob = 0.0
    cont.debug_vis = False
    return
  # Curriculum B: fewer crates + relocate grasp under rim for clearable approaches.
  cont.presence_prob = float(cfg.curriculum_b_presence_prob)
  cont.presence_requires_reach = True
  cont.presence_requires_need_waist = True
  cont.presence_also_on_low_z = True
  cont.relocate_grasp_when_present = bool(cfg.relocate_grasp_when_present)
  cont.debug_vis = False
  # Stronger IK-layer lift/repulsion while collecting under crates.
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
  """Keep the 3 spheres for the controlled arm (L:0:3, R:3:6)."""
  return clr6[:, :3] if arm_left else clr6[:, 3:6]


def meters_to_ee_raw(
  delta_m: torch.Tensor, scale: tuple[float, float, float]
) -> torch.Tensor:
  """Invert ``delta = scale * tanh(raw)`` for EeResidualIkAction."""
  s = torch.tensor(scale, device=delta_m.device, dtype=delta_m.dtype)
  x = torch.clamp(delta_m / s.clamp_min(1e-6), -0.999, 0.999)
  return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def _scalarize_log_val(v) -> float:
  if isinstance(v, torch.Tensor):
    return float(v.detach().float().mean().cpu())
  return float(v)


def _aggregate_ep_logs(ep_logs: list[dict]) -> dict[str, float]:
  """Mean per-term PPO episode rewards (monitor only; not used for BPTT)."""
  if not ep_logs:
    return {}
  keys = {k for ep in ep_logs for k in ep}
  out: dict[str, float] = {}
  for key in sorted(keys):
    vals = [_scalarize_log_val(ep[key]) for ep in ep_logs if key in ep]
    if vals:
      # Keep Homie/mjlab key shape; prefix so JSON is easy to filter.
      out[f"ppo/{key}"] = sum(vals) / len(vals)
  return out


class Stage3SimCollector:
  """Phase-3 env + frozen phase-2 HIM loco + EE attractor/actor."""

  def __init__(
    self,
    cfg: Stage3BpttCfg,
    *,
    stage2_checkpoint: str,
    num_envs: int = 64,
    ee_mode: Literal["attractor", "actor"] = "attractor",
    actor: Stage3Actor | None = None,
  ):
    self.cfg = cfg
    self.device = torch.device(cfg.device)
    self.ee_mode = ee_mode
    self.arm_left = cfg.arm_side == "left"
    self.actor = actor

    env_cfg = load_env_cfg("Mjlab-Homie-Flat-Unitree-G1-Phase3")
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(cfg.seed)
    _configure_curriculum(env_cfg, cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(self.device))
    self.vec = RslRlVecEnvWrapper(raw_env)
    self.env = self.vec.unwrapped

    rl_cfg = load_rl_cfg("Mjlab-Homie-Flat-Unitree-G1-Phase3")
    train_cfg = asdict(rl_cfg)

    self.runner = HomieOnPolicyRunner(
      self.vec, train_cfg, log_dir=None, device=str(self.device)
    )
    if not stage2_checkpoint:
      raise ValueError("stage2_checkpoint is required for frozen loco")
    path = Path(stage2_checkpoint)
    if not path.is_file():
      raise FileNotFoundError(path)
    self.runner.load(str(path), strict=False)
    self.runner._maybe_setup_teacher()
    self.runner.alg.actor_critic.eval()
    for p in self.runner.alg.actor_critic.parameters():
      p.requires_grad_(False)
    self.policy = self.runner.get_inference_policy(device=str(self.device))

    # EE scale from action cfg (meters after tanh).
    ee_cfg = self.env.action_manager.get_term("ee_residual").cfg
    self.ee_scale = tuple(float(x) for x in ee_cfg.ee_delta_scale)

    robot = self.env.scene["robot"]
    self.n_joints = int(robot.data.joint_pos.shape[-1])
    self._state0 = pack_robot_state(self.env, arm_left=self.arm_left)
    self.state_dim = int(self._state0.shape[-1])
    self.control_dim = self.n_joints
    self.goal_dim = 3
    self.geom_dim = 10
    self.obs_dim = 3 + 3 + 3 + cfg.sdf_query_count + 6  # palm, goal, last_a, sdf_phi, clearance
    # If include_valid doubles sdf — we use phi only for actor obs v1.
    self.feat_dim = self.state_dim + self.goal_dim + self.geom_dim

    self.last_a = torch.zeros(self.env.num_envs, 3, device=self.device)
    self._env_ep_id = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.long
    )

  def _force_reach_flags(self) -> None:
    grasp = self.env.command_manager.get_term("grasp")
    grasp.is_target[:] = 1.0
    grasp.is_waist[:] = 1.0
    grasp.waist_on = True

  def _build_actor_obs(self) -> torch.Tensor:
    palm = palm_pos_base(self.env, arm_left=self.arm_left)
    goal, _ = goal_geom_from_env(self.env)
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
    return torch.cat((palm, goal, self.last_a, sdf, clr), dim=-1)

  def _ee_delta_m(self) -> torch.Tensor:
    if self.ee_mode == "actor" and self.actor is not None:
      obs = self._build_actor_obs()
      with torch.no_grad():
        delta = self.actor(obs)
    else:
      delta = attractor_delta(
        self.env,
        arm_left=self.arm_left,
        scale=min(self.cfg.action_scale_m, 0.015),
        clip=self.cfg.action_scale_m,
      )
    # Curriculum B: attractor/actor residual also gets IK crate bias so
    # collection does not plow through walls before the actor learns.
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

  def _applied_joint_targets(self) -> torch.Tensor:
    """Full PD position command vector (n_joints), matching sim apply."""
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
    obs_td = self.vec.get_observations()
    n = self.env.num_envs
    coll_env_sum = 0.0
    present_sum = 0.0
    rew_sum = 0.0
    ppo_step_rew_sum = 0.0
    # PPO episode-term logs (extras["log"] on reset) — monitor loco health only.
    ep_logs: list[dict] = []
    for step_i in range(int(num_steps)):
      state = pack_robot_state(self.env, arm_left=self.arm_left)
      goal, geom = goal_geom_from_env(self.env)
      with torch.inference_mode():
        full_a = self.policy(obs_td)
      delta_m = self._ee_delta_m()
      ee_raw = meters_to_ee_raw(delta_m, self.ee_scale)
      act = full_a.clone()
      if act.shape[-1] >= 17:
        act[:, 14:17] = ee_raw
      elif act.shape[-1] == 14:
        act = torch.cat((act, ee_raw), dim=-1)

      obs_td, ppo_rew, dones, extras = self.vec.step(act)
      self._force_reach_flags()
      control = self._applied_joint_targets()
      next_state = pack_robot_state(self.env, arm_left=self.arm_left)

      palm = palm_pos_base(self.env, arm_left=self.arm_left)
      try:
        clr6 = arm_clearance_field(self.env, d_max=self.cfg.d_max)
      except Exception:  # noqa: BLE001
        clr6 = torch.ones(n, 6, device=self.device) * self.cfg.d_max
      # Only the controlled arm — idle-side spheres must not dominate safe loss.
      clr = _active_arm_clearance(clr6, arm_left=self.arm_left)
      try:
        present = self.env.command_manager.get_term("container").present
      except Exception:  # noqa: BLE001
        present = torch.zeros(n, device=self.device)
      r = stage3_reward(
        palm_pos=palm,
        goal_pos=goal,
        clearance=clr,
        action=delta_m,
        last_action=self.last_a,
        sigma_p=self.cfg.sigma_p,
        d_safe=self.cfg.d_safe,
        d_max=self.cfg.d_max,
        w_target=self.cfg.w_target,
        w_safe=self.cfg.w_safe,
        w_smooth=self.cfg.w_smooth,
        present=present,
      )
      dones_f = dones.float()
      truncated = torch.zeros_like(dones_f)
      if isinstance(extras, dict) and "time_outs" in extras:
        truncated = extras["time_outs"].float()
      terminated = (dones_f - truncated).clamp(min=0.0)
      # Monitor-only: mjlab emits Episode_Reward/* in extras["log"] on episode end.
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
        env_id=env_ids,
        episode_id=self._env_ep_id,
      )
      self.last_a = delta_m.detach()
      done_mask = dones.bool()
      if bool(done_mask.any()):
        self._env_ep_id[done_mask] += 1
        self.last_a[done_mask] = 0.0
        if hasattr(self.policy, "reset"):
          self.policy.reset()
      rew_sum += float(r.mean().item())
      ppo_step_rew_sum += float(ppo_rew.mean().item())
      # Env-mean penetration of the *active* arm (not "any of 4096 envs").
      hit = (clr.min(dim=-1).values < 0.0) & (present > 0.5)
      coll_env_sum += float(hit.float().mean().item())
      present_sum += float((present > 0.5).float().mean().item())

    out = {
      "collect_steps": float(num_steps),
      "collect_rew_mean": rew_sum / max(num_steps, 1),
      "ppo_step_rew_mean": ppo_step_rew_sum / max(num_steps, 1),
      "collect_neg_clearance_frac": coll_env_sum / max(num_steps, 1),
      "collect_present_frac": present_sum / max(num_steps, 1),
      "replay_size": float(replay.size),
      "ppo_ep_log_count": float(len(ep_logs)),
    }
    out.update(_aggregate_ep_logs(ep_logs))
    return out

  def make_fns(self):
    """Callables for Stage3BpttTrainer imagination (v1 open-loop loco)."""

    scale = self.ee_scale
    arm_left = self.arm_left
    n_j = self.n_joints
    device = self.device
    jp = self.env.action_manager.get_term("joint_pos")
    ee = self.env.action_manager.get_term("ee_residual")
    robot = self.env.scene["robot"]
    loco_ids = jp.target_ids
    arm_ids = ee._left_ids if arm_left else ee._right_ids

    def control_fn(state, delta_p, goal, geom):
      # Open-loop: hold current default + write arm via identity (Δp embedded).
      # Full DLS-from-state imagination lands in a follow-up; v1 keeps WM
      # learning from real controls while BPTT still sees delta_p→reward path
      # through geom_fn on predicted palm (state tail).
      del goal, geom
      b = state.shape[0]
      u = robot.data.default_joint_pos[:1].expand(b, -1).clone()
      # Encode Δp into a few unused channels? Prefer concatenating is wrong dim.
      # Instead keep last real control structure: arm joints unchanged here;
      # WM was trained on real DLS controls. Actor still gets reward via palm.
      u = u.clone()
      # Mildly bias arm joints with delta for a weak differentiable path.
      k = min(3, arm_ids.numel())
      u[:, arm_ids[:k]] = u[:, arm_ids[:k]] + delta_p[:, :k]
      return u

    d_max = float(self.cfg.d_max)
    sdf_k = int(self.cfg.sdf_query_count)

    def obs_fn(state, goal, geom, last_a):
      palm = state[..., -3:]
      # Analytic SDF/clearance from packed geom (not fake "always safe").
      if geom is not None and geom.shape[-1] >= 10:
        sdf = sdf_query_from_geom(geom, query_count=sdf_k, d_max=d_max)
        clr3 = palm_probe_clearance(palm, geom, d_max=d_max)
        pad = torch.ones(
          state.shape[0], 3, device=state.device, dtype=state.dtype
        )
        clr = (
          torch.cat((clr3, pad), dim=-1)
          if arm_left
          else torch.cat((pad, clr3), dim=-1)
        )
      else:
        sdf = torch.zeros(state.shape[0], sdf_k, device=state.device)
        clr = torch.ones(state.shape[0], 6, device=state.device) * d_max
      return torch.cat((palm, goal[..., :3], last_a, sdf, clr), dim=-1)

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

    return control_fn, obs_fn, feat_fn, geom_fn
