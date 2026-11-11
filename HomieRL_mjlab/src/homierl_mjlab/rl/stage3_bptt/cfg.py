"""Stage-3 BPTT training configuration (engineering defaults, not claimed optima)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Checkpoint / replay gate — refuse silent load of old 3-D hier actors.
ACTION_SEMANTICS_BODY_EE = "body14_ee3_palm_delta_v1"
ACTION_SEMANTICS_HIER = "hier_ee0_offset_v0"
ACTION_SEMANTICS_RESIDUAL = "ee_residual_v0"


@dataclass
class Stage3BpttCfg:
  # --- control ---
  # body_ee_palm_delta (default): Actor [a_body(14), a_ee(3)];
  #   p_sub_B = palm_B + delta_B; frozen phase-2 teacher supervises body.
  # hierarchical_subgoal (legacy): offset from episode p_ee0; frozen loco policy.
  # ee_residual (legacy): Δp on EE IK only.
  control_mode: str = "body_ee_palm_delta"
  action_semantics: str = ACTION_SEMANTICS_BODY_EE
  arm_side: str = "right"  # "left" | "right"
  body_dim: int = 14
  ee_dim: int = 3
  # Homie joint-pos action scale: q = q_default + scale * a_body.
  body_action_scale: float = 0.25
  # EE palm-delta |Δp| per axis (m) after tanh.
  ee_delta_scale_m: float = 0.03
  # Max |p_sub_W| change per policy step (m) after base→world.
  subgoal_rate_limit_m: float = 0.04
  # Frozen phase-2 teacher MSE weight on a_body (regularizer, not the only loco signal).
  w_teacher: float = 2.0
  # Blend Phase-1/2 env PPO step reward into Stage3 replay / critic targets.
  # This is the same Episode_Reward/* stack (tracking, orientation, height, …).
  w_loco: float = 1.0
  # Body curriculum: execute teacher loco in sim (Actor only learns; no bad a_body on robot).
  body_exec_teacher: bool = True
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
