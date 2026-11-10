"""Stage-3 BPTT training configuration (engineering defaults, not claimed optima)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage3BpttCfg:
  # --- control ---
  # ee_residual: actor Δp added on EE IK (legacy Stage-3).
  # hierarchical_subgoal: actor offset from episode EE anchor → grasp target;
  #   frozen phase-2 loco + EE IK chase that short-horizon target (套娃).
  control_mode: str = "hierarchical_subgoal"
  arm_side: str = "right"  # "left" | "right"
  action_scale_m: float = 0.02  # legacy residual |Δp| per step (m)
  # Hierarchical: max |offset| from p_ee0 in world meters (must cover goal−ee0).
  subgoal_offset_scale_m: float = 0.55
  # Hierarchical: max |p_sub| change per policy step (m).
  subgoal_rate_limit_m: float = 0.04
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
  # Safe ≥ target, but not so large that constant penetration kills all gradients.
  w_safe: float = 1.5
  # Dense clearance-above-margin bonus (guides away before collision).
  w_margin: float = 0.5
  w_smooth: float = 0.05
  w_stability: float = 0.0

  # --- world model ---
  wm_hidden: tuple[int, ...] = (512, 512, 256)
  wm_lr: float = 3e-4  # lower than 1e-3; residual WM blows up less often
  wm_window: int = 8
  wm_batch: int = 256
  wm_warmup_steps: int = 2000
  # Clamp per-step residual Δs predicted by WM (joint/root units mixed).
  wm_residual_clip: float = 0.5
  # Huber δ for WM fit (more robust than plain MSE on outliers).
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
  # Clamp imagined stage3 reward before BPTT (avoid J → ±inf).
  reward_clip: float = 5.0
  # Skip optimizer step / restore weights when loss or grads are non-finite.
  nan_guard: bool = True
  # After this many consecutive NaN train iters, reload last-good + shrink LRs.
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

  # paths (fill at launch)
  stage2_checkpoint: str = ""
  log_dir: str = "logs/stage3_bptt"
  # A=no container (reach only); B=crate present for SDF / clearance BPTT.
  curriculum: str = "B"
  # Start milder than 0.7 so replay mixes free reaches with crates.
  curriculum_b_presence_prob: float = 0.35
  # Place grasp under open-top so lift-over is feasible (Homie crate curriculum).
  relocate_grasp_when_present: bool = True
  avoid_lift_margin: float = 0.12
  avoid_repulsion_gain: float = 3.0
  avoid_xy_slow: float = 0.10

  extra: dict = field(default_factory=dict)
