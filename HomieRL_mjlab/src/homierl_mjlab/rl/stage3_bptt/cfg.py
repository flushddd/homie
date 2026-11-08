"""Stage-3 BPTT training configuration (engineering defaults, not claimed optima)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage3BpttCfg:
  # --- control ---
  arm_side: str = "right"  # "left" | "right"
  action_scale_m: float = 0.02  # max |Δp| per policy step (m)
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
  # Safe ≥ target so rim collisions are not ignored for reach progress.
  w_safe: float = 2.5
  w_smooth: float = 0.05
  w_stability: float = 0.0

  # --- world model ---
  wm_hidden: tuple[int, ...] = (512, 512, 256)
  wm_lr: float = 1e-3
  wm_window: int = 8
  wm_batch: int = 256
  wm_warmup_steps: int = 2000

  # --- actor / critic ---
  actor_hidden: tuple[int, ...] = (256, 256, 128)
  critic_hidden: tuple[int, ...] = (256, 256, 128)
  actor_lr: float = 3e-4
  critic_lr: float = 1e-3
  gamma: float = 0.99
  lam: float = 0.95
  imagine_horizon: int = 8
  grad_clip: float = 1.0
  target_critic_tau: float = 0.005

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
  curriculum_b_presence_prob: float = 0.70

  extra: dict = field(default_factory=dict)
