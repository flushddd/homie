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
    }
  )


def homie_g1_him_runner_cfg() -> HomieRlOnPolicyRunnerCfg:
  return HomieRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
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
      learning_rate=3e-4,
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      schedule="adaptive",
    ),
    experiment_name="homie_g1",
    max_iterations=100_000,
    num_steps_per_env=24,
    save_interval=200,
    logger="tensorboard",
    upload_model=False,
    wandb_project="homie_mjlab",
    him={
      "use_flip": True,
      "symmetry_scale": 0.5,  # was 1.0; less pressure while loco is unstable
      "actor_history_length": 6,
      "critic_history_length": 1,
      "max_learning_rate": 1e-3,  # prevent adaptive LR → 0.01 blow-up
      "min_learning_rate": 1e-5,
    },
  )
