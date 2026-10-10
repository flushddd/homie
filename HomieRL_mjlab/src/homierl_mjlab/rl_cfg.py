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
    }
  )


def homie_g1_him_runner_cfg() -> HomieRlOnPolicyRunnerCfg:
  """Align with Homie ``G1_Inspire_Waist_RoughCfgPPO`` / ``LeggedRobotCfgPPO``."""
  return HomieRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # Homie ``LeggedRobotCfgPPO.policy.actor_hidden_dims``
      hidden_dims=(512, 256, 256),
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 256)),
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
    experiment_name="homie_g1",
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
    },
  )
