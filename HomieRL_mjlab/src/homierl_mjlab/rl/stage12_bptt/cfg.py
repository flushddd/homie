"""Stage-1/2 SuperMimic BPTT config (experimental; parallel to HIM-PPO)."""

from __future__ import annotations

from dataclasses import dataclass, field

ACTION_SEMANTICS_P1 = "loco12_phase1_v0"
ACTION_SEMANTICS_P2 = "loco14_phase2_v0"

TASK_PHASE1 = "Mjlab-Homie-Flat-Unitree-G1"
TASK_PHASE2 = "Mjlab-Homie-Flat-Unitree-G1-Phase2"


@dataclass
class Stage12BpttCfg:
  # 1 = stand/walk legs; 2 = legs + waist roll/pitch + grasp cmd.
  phase: int = 1
  body_dim: int = 12  # overwritten from phase in resolve_dims()
  body_action_scale: float = 0.25

  # Optional frozen phase-1 BPTT/PPO teacher for phase-2 BC (not Stage-3 HIM).
  phase1_checkpoint: str = ""
  w_teacher: float = 1.0
  # Real collect: blend env PPO step reward (Phase-1/2 Episode_Reward stack).
  w_loco: float = 1.0

  # Soft imag proxy (tilt); rich loco terms come from real ppo_rew.
  w_stability: float = 0.15
  stability_limit_rad: float = 1.047
  stability_soft_rad: float = 0.25
  imag_soft_survival: bool = True
  reward_clip: float = 5.0
  clip_critic_reward: bool = True
  clip_imag_reward: bool = True

  wm_hidden: tuple[int, ...] = (512, 512, 256)
  wm_lr: float = 3e-4
  wm_window: int = 8
  wm_batch: int = 256
  wm_residual_clip: float = 0.5
  wm_huber_delta: float = 1.0

  actor_hidden: tuple[int, ...] = (256, 256, 128)
  critic_hidden: tuple[int, ...] = (256, 256, 128)
  actor_lr: float = 1e-4
  critic_lr: float = 3e-4
  gamma: float = 0.99
  lam: float = 0.95
  imagine_horizon: int = 8
  grad_clip: float = 1.0
  target_critic_tau: float = 0.005
  nan_guard: bool = True
  nan_reload_patience: int = 3
  nan_lr_decay: float = 0.5

  replay_capacity: int = 200_000
  collect_steps_per_iter: int = 2048
  wm_updates_per_iter: int = 4
  critic_updates_per_iter: int = 4
  actor_updates_per_iter: int = 2
  seed: int = 0
  device: str = "cuda:0"
  log_dir: str = "logs/stage12_bptt"

  action_semantics: str = ACTION_SEMANTICS_P1
  extra: dict = field(default_factory=dict)

  def resolve_dims(self) -> None:
    p = int(self.phase)
    if p <= 1:
      self.phase = 1
      self.body_dim = 12
      self.action_semantics = ACTION_SEMANTICS_P1
    else:
      self.phase = 2
      self.body_dim = 14
      self.action_semantics = ACTION_SEMANTICS_P2

  def task_id(self) -> str:
    return TASK_PHASE1 if int(self.phase) <= 1 else TASK_PHASE2
