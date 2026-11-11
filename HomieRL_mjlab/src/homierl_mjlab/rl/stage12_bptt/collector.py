"""Real-sim collector for Stage-1/2 SuperMimic BPTT (env PPO rewards)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from homierl_mjlab.rl.him_runner import HomieOnPolicyRunner
from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
from homierl_mjlab.rl.stage12_bptt.env_bridge import (
  flatten_policy_obs,
  pack_robot_state,
)
from homierl_mjlab.rl.stage12_bptt.networks import LocoActor
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.rewards_stage3 import (
  soft_stability_reward,
  tilt_angle_from_root_quat,
)

if TYPE_CHECKING:
  pass


class Stage12SimCollector:
  """Phase-1 or Phase-2 mjlab env + Stage12 loco actor."""

  def __init__(
    self,
    cfg: Stage12BpttCfg,
    *,
    num_envs: int = 64,
    actor: LocoActor | None = None,
    play: bool = False,
  ):
    self.cfg = cfg
    cfg.resolve_dims()
    self.device = torch.device(cfg.device)
    self.actor = actor
    self.play = bool(play)
    self.phase = int(cfg.phase)
    self.body_dim = int(cfg.body_dim)
    self.task_id = cfg.task_id()

    env_cfg = load_env_cfg(self.task_id, play=self.play)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(cfg.seed)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=str(self.device))
    self.vec = RslRlVecEnvWrapper(raw_env)
    self.env = self.vec.unwrapped

    # Optional phase-1 teacher for phase-2 BC (same task dims only if phase==1 ckpt
    # remapped — scaffold loads phase-1 HIM into a frozen inference policy when path set).
    self.teacher_policy = None
    if self.phase >= 2 and str(cfg.phase1_checkpoint).strip():
      self._load_phase1_teacher(str(cfg.phase1_checkpoint))

    robot = self.env.scene["robot"]
    self.n_joints = int(robot.data.joint_pos.shape[-1])
    self._state0 = pack_robot_state(self.env)
    self.state_dim = int(self._state0.shape[-1])
    self.control_dim = self.n_joints
    self.goal_dim = 3  # placeholder cmd (zeros) for WM ctx layout
    self.geom_dim = 1  # unused pad
    obs0 = flatten_policy_obs(self.vec.get_observations())
    self.obs_dim = int(obs0.shape[-1])
    self.feat_dim = self.state_dim + self.goal_dim + self.geom_dim

    jp = self.env.action_manager.get_term("joint_pos")
    self.loco_ids = jp.target_ids
    self.env_action_dim = int(self.env.action_manager.total_action_dim)
    if self.body_dim > self.env_action_dim:
      raise RuntimeError(
        f"body_dim={self.body_dim} > env action_dim={self.env_action_dim}"
      )

    self.last_a = torch.zeros(self.env.num_envs, self.body_dim, device=self.device)
    self._env_ep_id = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.long
    )
    self._ep_len = torch.zeros(self.env.num_envs, device=self.device)
    self._last_mean_ep_len = 0.0

  def _load_phase1_teacher(self, path: str) -> None:
    """Best-effort: load phase-1 PPO HIM for BC targets (14-D padded with 0 waist)."""
    p = Path(path)
    if not p.is_file():
      raise FileNotFoundError(path)
    # Teacher runs on a phase-1-sized action head; we only take first 12 dims.
    rl_cfg = load_rl_cfg("Mjlab-Homie-Flat-Unitree-G1")
    train_cfg = asdict(rl_cfg)
    # Reuse same vec — phase-2 obs is larger; runner load remaps when possible.
    runner = HomieOnPolicyRunner(
      self.vec, train_cfg, log_dir=None, device=str(self.device)
    )
    runner.load(str(p), strict=False)
    runner.alg.actor_critic.eval()
    for param in runner.alg.actor_critic.parameters():
      param.requires_grad_(False)
    self.teacher_policy = runner.get_inference_policy(device=str(self.device))
    self._teacher_runner = runner  # keep alive

  def _build_obs(self) -> torch.Tensor:
    return flatten_policy_obs(self.vec.get_observations())

  def _actor_body(self) -> torch.Tensor:
    n = self.env.num_envs
    if self.actor is None:
      return torch.zeros(n, self.body_dim, device=self.device)
    obs = self._build_obs()
    with torch.no_grad():
      return self.actor(obs)

  def _teacher_body(self, obs_td) -> torch.Tensor | None:
    if self.teacher_policy is None:
      return None
    with torch.inference_mode():
      full = self.teacher_policy(obs_td)
    # Phase-1 teacher: 12-D; pad / trim to body_dim.
    t = full[..., : min(full.shape[-1], self.body_dim)]
    if t.shape[-1] < self.body_dim:
      pad = torch.zeros(
        t.shape[0], self.body_dim - t.shape[-1], device=t.device, dtype=t.dtype
      )
      t = torch.cat((t, pad), dim=-1)
    return t.detach()

  def _applied_joint_targets(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    u = robot.data.default_joint_pos.clone()
    jp = self.env.action_manager.get_term("joint_pos")
    tids = jp.target_ids
    u[:, tids] = jp._processed_actions - robot.data.encoder_bias[:, tids]
    return u

  def collect(self, replay: EpisodeReplay, num_steps: int) -> dict[str, float]:
    obs_td = self.vec.get_observations()
    n = self.env.num_envs
    rew_sum = 0.0
    ppo_sum = 0.0
    fall_sum = 0.0
    teacher_mse_sum = 0.0
    body_norm_sum = 0.0
    finished_ep_lens: list[float] = []

    for _ in range(int(num_steps)):
      state = pack_robot_state(self.env)
      a_body = self._actor_body()
      teacher = self._teacher_body(obs_td)
      if teacher is not None:
        teacher_mse_sum += float(torch.mean((a_body - teacher) ** 2).item())

      act = torch.zeros(n, self.env_action_dim, device=self.device)
      act[:, : self.body_dim] = a_body

      obs_td, ppo_rew, dones, extras = self.vec.step(act)
      control = self._applied_joint_targets()
      next_state = pack_robot_state(self.env)

      ppo = ppo_rew.reshape(n).float()
      # Soft stability bonus on real steps (same formula as Stage-3 helper).
      # state = q|qd|pos3|quat4|lin3|ang3 → quat at [2*nj+3 : 2*nj+7]
      nj = self.n_joints
      quat = state[..., 2 * nj + 3 : 2 * nj + 7]
      tilt = tilt_angle_from_root_quat(quat)
      stab = soft_stability_reward(
        tilt,
        limit_rad=self.cfg.stability_limit_rad,
        soft_rad=self.cfg.stability_soft_rad,
      )
      w_loco = float(self.cfg.w_loco)
      w_stab = float(self.cfg.w_stability)
      r = w_loco * ppo + w_stab * stab

      dones_f = dones.float()
      truncated = torch.zeros_like(dones_f)
      if isinstance(extras, dict) and "time_outs" in extras:
        truncated = extras["time_outs"].float()
      terminated = (dones_f - truncated).clamp(min=0.0)
      fall_sum += float(terminated.mean().item())

      goal = torch.zeros(n, self.goal_dim, device=self.device)
      geom = torch.zeros(n, self.geom_dim, device=self.device)
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
        present=torch.zeros(n, device=self.device),
        env_id=env_ids,
        episode_id=self._env_ep_id,
        actor_action=a_body.detach(),
        teacher_body=teacher,
      )
      self.last_a = a_body.detach()
      body_norm_sum += float(a_body.norm(dim=-1).mean().item())
      self._ep_len += 1.0
      done_mask = dones.bool()
      if bool(done_mask.any()):
        finished_ep_lens.extend(self._ep_len[done_mask].detach().cpu().tolist())
        self._ep_len[done_mask] = 0.0
        self._env_ep_id[done_mask] += 1
        self.last_a[done_mask] = 0.0
        if self.teacher_policy is not None and hasattr(self.teacher_policy, "reset"):
          self.teacher_policy.reset()
      rew_sum += float(r.mean().item())
      ppo_sum += float(ppo.mean().item())

    denom = max(num_steps, 1)
    out = {
      "collect_steps": float(num_steps),
      "collect_rew_mean": rew_sum / denom,
      "ppo_step_rew_mean": ppo_sum / denom,
      "fall_frac": fall_sum / denom,
      "actor_body_norm": body_norm_sum / denom,
      "replay_size": float(replay.size),
      "phase": float(self.phase),
      "body_dim": float(self.body_dim),
      "w_loco": float(self.cfg.w_loco),
    }
    if self.teacher_policy is not None:
      out["teacher_mse_collect"] = teacher_mse_sum / denom
    if finished_ep_lens:
      self._last_mean_ep_len = float(sum(finished_ep_lens) / len(finished_ep_lens))
      out["mean_episode_length"] = self._last_mean_ep_len
      out["episode_length_count"] = float(len(finished_ep_lens))
    else:
      out["mean_episode_length"] = float(self._last_mean_ep_len)
      out["episode_length_count"] = 0.0
    return out

  def max_episode_length_steps(self) -> float:
    try:
      ep_s = float(getattr(self.env.cfg, "episode_length_s", 0.0) or 0.0)
      dt = float(self.env.step_dt)
      if ep_s > 0.0 and dt > 0.0:
        return max(ep_s / dt, 1.0)
    except Exception:  # noqa: BLE001
      pass
    return 1000.0

  def make_fns(self):
    """Imagination helpers (loco proxy; rich rewards stay on real collect)."""
    n_j = self.n_joints
    loco_ids = self.loco_ids
    body_dim = self.body_dim
    body_scale = float(self.cfg.body_action_scale)
    default_q = self.env.scene["robot"].data.default_joint_pos[0].detach().clone()
    obs_dim = self.obs_dim

    def control_fn(state, action, goal, geom):
      del goal, geom
      u = state[..., :n_j].clone()
      a = action[..., :body_dim]
      def_q = default_q.to(device=u.device, dtype=u.dtype).unsqueeze(0).expand_as(u)
      u[:, loco_ids] = def_q[:, loco_ids] + body_scale * a
      return u

    def obs_fn(state, goal, geom, last_a):
      # Compact imag obs: root slice + last action (pad/trim to obs_dim).
      del goal, geom
      b = state.shape[0]
      root = state[..., -13:]  # pos3 quat4 lin3 ang3
      la = last_a
      if la.shape[-1] != body_dim:
        pad = torch.zeros(b, body_dim, device=state.device, dtype=state.dtype)
        n = min(body_dim, la.shape[-1])
        pad[..., :n] = la[..., :n]
        la = pad
      raw = torch.cat((root, la), dim=-1)
      if raw.shape[-1] < obs_dim:
        z = torch.zeros(b, obs_dim - raw.shape[-1], device=state.device, dtype=state.dtype)
        return torch.cat((raw, z), dim=-1)
      return raw[..., :obs_dim]

    def feat_fn(state, goal, geom):
      return torch.cat((state, goal, geom), dim=-1)

    def geom_fn(state, goal, geom):
      del goal, geom
      # Dummy palm/clearance placeholders for API parity.
      palm = state[..., -3:]
      clr = torch.ones(state.shape[0], 3, device=state.device) * 0.4
      return palm, clr

    return control_fn, obs_fn, feat_fn, geom_fn
