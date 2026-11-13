"""Stage-3 BPTT training configuration (engineering defaults, not claimed optima)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Checkpoint / replay gate — refuse silent load of old actors / obs layouts.
# v4: raw a_body + HIM-aligned 6-frame loco history for body_net (same as teacher).
ACTION_SEMANTICS_BODY_EE = "body14_ee3_palm_delta_v4_loco_hist6"
ACTION_SEMANTICS_HIER = "hier_ee0_offset_v0"
ACTION_SEMANTICS_RESIDUAL = "ee_residual_v0"


@dataclass
class Stage3BpttCfg:
  # --- control ---
  # body_ee_palm_delta (default): Actor [a_body(14), a_ee(3)];
  #   body ← distill frozen phase-2/3 HIM on loco obs;
  #   a_ee ← palm Δp command injected as p_sub for loco+IK.
  # hierarchical_subgoal (legacy): offset from episode p_ee0; frozen loco policy.
  # ee_residual (legacy): Δp on EE IK only.
  control_mode: str = "body_ee_palm_delta"
  action_semantics: str = ACTION_SEMANTICS_BODY_EE
  # Concat Phase-3 env actor (HIM) obs in front of reach features for body head.
  include_loco_obs: bool = True
  # Match Homie / HIM ``actor_history_length``: body_net sees stacked one-step
  # loco frames (same entry as teacher ``act_inference``), not a single frame.
  loco_history_length: int = 6
  arm_side: str = "right"  # "left" | "right"
  body_dim: int = 14
  ee_dim: int = 3
  # Homie joint-pos action scale: q = q_default + scale * a_body (phase-1/2).
  body_action_scale: float = 0.25
  # EE palm-delta hard clip |Δp| per axis (m); raw meters, no tanh.
  # Injected as p_sub (oracle-style); env EE residual stays off.
  ee_delta_scale_m: float = 0.03
  # Max |p_sub_W| change per policy step (m) after base→world.
  subgoal_rate_limit_m: float = 0.04
  # Frozen phase-2 teacher MSE weight on a_body (regularizer, not the only loco signal).
  w_teacher: float = 2.0
  # a_body matches phase-1/2 HIM means: raw (no tanh), numeric clip ±100
  # (Homie ``normalization.clip_actions``). Teacher uses the same space for BC.
  #   "none" / "raw": identity + ±clip (default; matches phase-1/2)
  #   "clip": hard clamp only
  #   "tanh": legacy soft map (v1/v2) — do not use for new trains
  teacher_action_norm: str = "none"
  teacher_action_clip: float = 100.0
  # Hard clamp on Stage3 a_body before exec / BC (0 = off). Same as clip_actions.
  body_action_clip: float = 100.0
  # Blend Phase-1/2 env PPO step reward into Stage3 replay / critic targets.
  # This is the same Episode_Reward/* stack (tracking, orientation, height, …).
  w_loco: float = 1.0
  # Body curriculum: execute teacher loco in sim (Actor only learns; no bad a_body on robot).
  body_exec_teacher: bool = True
  # While arm locked + body_exec_teacher: fraction of actor mixed into sim exec
  # (0 = pure teacher / never takeoff; >0 lets a_body see loco reward gradually).
  body_exec_mix: float = 0.0
  # Hard trust region around teacher for any actor component in exec_body:
  #   exec = teacher + clip(blend - teacher, ±trust). Prevents fly-away when
  #   a_body is wrong (play/train). 0 disables. Keep >0 after arm unlock too.
  body_trust_region: float = 0.20
  # Soft cap on ||a_body|| before blend (raw HIM coords); 0 = off.
  body_max_norm: float = 0.0
  # Legacy scales (kept for hierarchical / residual ablation).
  action_scale_m: float = 0.02
  subgoal_offset_scale_m: float = 0.55
  policy_dt: float = 0.02
  dls_damping: float = 0.08
  max_joint_delta: float = 0.05
  lock_wrist: bool = True

  # --- obs / geometry ---
  sdf_query_count: int = 64
  d_max: float = 0.40
  d_safe: float = 0.05
  sigma_p: float = 0.08
  reach_threshold: float = 0.03
  reach_hold_steps: int = 5

  # --- rewards ---
  w_target: float = 1.0
  w_approach: float = 0.3
  sigma_far: float = 0.35
  w_safe: float = 1.5
  w_margin: float = 0.5
  w_smooth: float = 0.05
  w_soft_success: float = 0.2
  w_stability: float = 0.15
  stability_limit_rad: float = 1.047
  stability_soft_rad: float = 0.25
  imag_soft_survival: bool = True
  use_palm_step: bool = True
  imag_use_present: bool = True
  clip_critic_reward: bool = True
  clip_imag_reward: bool = True

  # --- world model ---
  wm_hidden: tuple[int, ...] = (512, 512, 256)
  wm_lr: float = 3e-4
  wm_window: int = 8
  wm_batch: int = 256
  wm_warmup_steps: int = 2000
  wm_residual_clip: float = 0.5
  wm_huber_delta: float = 1.0

  # --- actor / critic ---
  actor_hidden: tuple[int, ...] = (256, 256, 128)
  critic_hidden: tuple[int, ...] = (256, 256, 128)
  actor_lr: float = 1e-4
  critic_lr: float = 3e-4
  gamma: float = 0.99
  lam: float = 0.95
  imagine_horizon: int = 8
  grad_clip: float = 1.0
  target_critic_tau: float = 0.005
  reward_clip: float = 5.0
  nan_guard: bool = True
  nan_reload_patience: int = 3
  nan_lr_decay: float = 0.5

  # --- loop ---
  replay_capacity: int = 200_000
  collect_steps_per_iter: int = 2048
  wm_updates_per_iter: int = 4
  critic_updates_per_iter: int = 4
  actor_updates_per_iter: int = 2
  seed: int = 0
  device: str = "cuda:0"

  stage2_checkpoint: str = ""
  log_dir: str = "logs/stage3_bptt"
  # Static A/B when curriculum_auto=False. With auto: starts body-only, no crates.
  curriculum: str = "A"
  curriculum_b_presence_prob: float = 0.35
  # Progressive: body (arm locked) → arm (no box) → box presence ramp.
  curriculum_auto: bool = True
  curr_ema_beta: float = 0.9
  curr_body_min_iters: int = 30
  curr_body_hold_iters: int = 5
  curr_teacher_mse_max: float = 0.08
  curr_ep_len_frac_min: float = 0.75
  curr_arm_min_iters: int = 40
  curr_arm_hold_iters: int = 5
  curr_reach_unlock_box: float = 0.50
  curr_box_min_iters_between_bumps: int = 20
  curr_box_hold_iters: int = 5
  curr_clear_reach_bump: float = 0.30
  curr_presence_step: float = 0.05
  curr_presence_max: float = 0.35
  relocate_grasp_when_present: bool = True
  avoid_lift_margin: float = 0.12
  avoid_repulsion_gain: float = 3.0
  avoid_xy_slow: float = 0.10

  extra: dict = field(default_factory=dict)

  def resolve_action_semantics(self) -> str:
    mode = str(self.control_mode)
    if mode == "body_ee_palm_delta":
      return ACTION_SEMANTICS_BODY_EE
    if mode == "hierarchical_subgoal":
      return ACTION_SEMANTICS_HIER
    if mode == "ee_residual":
      return ACTION_SEMANTICS_RESIDUAL
    return str(self.action_semantics)
