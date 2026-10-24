"""RL configuration for Homie G1 HIM-PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class HomieRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extends mjlab runner cfg with HIM-specific fields."""

  him: dict[str, Any] = field(
    default_factory=lambda: {
      "use_flip": True,
      "symmetry_scale": 1.0,
      "actor_history_length": 6,
      "critic_history_length": 1,
      "max_learning_rate": 1e-2,
      "min_learning_rate": 1e-5,
      "phase": 1,
      "use_teacher": False,
      "keep_loco_weight": 2.0,
      "keep_loco_dims": 12,
      "freeze_std_dims": 12,
      "stand_height": 0.74,
      "keep_height_tol": 0.02,
    }
  )


def homie_g1_him_runner_cfg(phase: int = 1) -> HomieRlOnPolicyRunnerCfg:
  """Align with Homie ``G1_Inspire_Waist_RoughCfgPPO`` / ``LeggedRobotCfgPPO``.

  ``phase=2``: teacher = phase-1; keep first 12 leg dims.
  ``phase=3``: teacher = phase-2; keep first 14 loco dims (legs+roll/pitch, no yaw).
  """
  phase = int(phase)
  use_teacher = phase >= 2
  # Oct-15 layouts: p2 is_waist @ 68+14+1=83; p3 @ 68+17+1=86
  if phase >= 3:
    keep_dims = 14
    run_name = "phase3"
    is_waist_idx = 86
    keep_always = True  # arm-reach keeps is_waist≈1; still lock 14 loco dims
    # Teacher is expanded p2 clone (same 229 net); pass full obs (keep locks means).
    teacher_one_step = 0
    keep_loco_weight = 5.0  # stronger than phase-2; new SDF/EE must not wreck loco
  elif phase == 2:
    keep_dims = 12
    run_name = "phase2"
    is_waist_idx = 83
    keep_always = False
    teacher_one_step = 0  # full student obs → teacher (Oct-15 / 2a2c0b1)
    keep_loco_weight = 2.0
  else:
    keep_dims = 12
    run_name = ""
    is_waist_idx = 83
    keep_always = False
    teacher_one_step = 0
    keep_loco_weight = 2.0
  return HomieRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Pre-alignment mjlab size (Homie paper cfg uses 512/256/256).
      hidden_dims=(512, 256, 128),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.01,
      learning_rate=1e-3,  # Homie LeggedRobotCfgPPO.algorithm
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      schedule="adaptive",
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
    ),
    # Same experiment dir so later phases can resume prior runs.
    experiment_name="homie_g1",
    run_name=run_name,
    max_iterations=100_000,
    num_steps_per_env=50,  # Homie G1_*CfgPPO.runner
    save_interval=200,
    # Homie ``normalization.clip_actions = 100``
    clip_actions=100.0,
    logger="tensorboard",
    upload_model=False,
    wandb_project="homie_mjlab",
    him={
      "use_flip": True,
      "symmetry_scale": 1.0,  # Homie algorithm.symmetry_scale
      "actor_history_length": 6,
      "critic_history_length": 1,
      # Homie adaptive LR clamps: min 1e-5, max 1e-2
      "max_learning_rate": 1e-2,
      "min_learning_rate": 1e-5,
      # Phase switch (env is separate via task registration).
      "phase": phase,
      "use_teacher": use_teacher,
      # Phase-2: lock 12 legs to phase-1. Phase-3: lock 14 loco (no waist yaw).
      "keep_loco_weight": keep_loco_weight,
      "keep_loco_dims": keep_dims,
      "freeze_std_dims": keep_dims,
      "stand_height": 0.74,
      "keep_height_tol": 0.02,
      "is_waist_one_step_idx": is_waist_idx,
      "keep_loco_always": keep_always,
      "teacher_one_step_obs": teacher_one_step,
    },
  )
