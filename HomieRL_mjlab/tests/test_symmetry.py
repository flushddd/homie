"""Symmetry flip sanity checks."""

import torch

from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic
from homierl_mjlab.rl.him.him_ppo import HIMPPO


def test_flip_actions_involution():
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu")
  x = torch.randn(8, 12)
  y = ppo.flip_actions(ppo.flip_actions(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_actor_obs_involution():
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  # Identity joint perm for smoke (no name map).
  ppo._joint_perm = list(range(29))
  ppo._joint_sign = [1.0] * 29
  x = torch.randn(4, 480)
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  # Commands/IMU/actions should round-trip; with identity perm joints too.
  assert torch.allclose(x, y, atol=1e-5)


if __name__ == "__main__":
  test_flip_actions_involution()
  test_flip_actor_obs_involution()
  print("ok")
