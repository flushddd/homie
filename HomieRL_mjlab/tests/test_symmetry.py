"""Symmetry flip sanity checks for unified Homie obs layout."""

import torch

from homierl_mjlab.mdp.observations import (
  HOMIE_ACTION_OBS_DIM,
  HOMIE_CLEARANCE_DIM,
  HOMIE_EE_ERR_DIM,
  HOMIE_GRASP_DIM,
  HOMIE_IS_WAIST_IDX,
  HOMIE_ONE_STEP_OBS,
  HOMIE_PROPRIO_DIM,
  HOMIE_SDF_DIM,
)
from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic
from homierl_mjlab.rl.him.him_ppo import HIMPPO


def test_unified_layout_constants():
  assert HOMIE_ONE_STEP_OBS == (
    HOMIE_PROPRIO_DIM
    + HOMIE_ACTION_OBS_DIM
    + HOMIE_GRASP_DIM
    + HOMIE_SDF_DIM
    + HOMIE_CLEARANCE_DIM
    + HOMIE_EE_ERR_DIM
  )
  assert HOMIE_ONE_STEP_OBS == 229
  assert HOMIE_IS_WAIST_IDX == 86


def test_flip_actions_involution():
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu")
  x = torch.randn(8, 12)
  y = ppo.flip_actions(ppo.flip_actions(x))
  assert torch.allclose(x, y, atol=1e-5)
  x17 = torch.randn(8, 17)
  y17 = ppo.flip_actions(ppo.flip_actions(x17))
  assert torch.allclose(x17, y17, atol=1e-5)


def test_flip_actor_obs_involution_legacy80():
  """Legacy loco-only one-step (68+12) still round-trips."""
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  ppo._joint_perm = list(range(29))
  ppo._joint_sign = [1.0] * 29
  x = torch.randn(4, 480)
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_actor_obs_involution_unified229():
  d = HOMIE_ONE_STEP_OBS
  hist = 6
  ac = HIMActorCritic(hist * d, d + 3, d, d + 3, hist, 1, 17)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  ppo._joint_perm = list(range(29))
  ppo._joint_sign = [1.0] * 29
  x = torch.randn(4, hist * d)
  # Discrete arm channel: 0 / 1 / 2
  g0 = HOMIE_PROPRIO_DIM + HOMIE_ACTION_OBS_DIM
  for t in range(hist):
    base = t * d + g0
    x[:, base + 6] = torch.tensor([0.0, 1.0, 2.0, 1.0])
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-4)
  assert HOMIE_IS_WAIST_IDX == g0 + 1


def test_flip_critic_lin_vel_at_tail():
  d = HOMIE_ONE_STEP_OBS
  dc = d + 3
  ac = HIMActorCritic(d, dc, d, dc, 1, 1, 17)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  ppo._joint_perm = list(range(29))
  ppo._joint_sign = [1.0] * 29
  x = torch.zeros(2, dc)
  x[:, -3:] = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
  y = ppo.flip_critic_obs(x)
  assert torch.allclose(y[:, -3], x[:, -3])
  assert torch.allclose(y[:, -2], -x[:, -2])
  assert torch.allclose(y[:, -1], x[:, -1])


if __name__ == "__main__":
  test_unified_layout_constants()
  test_flip_actions_involution()
  test_flip_actor_obs_involution_legacy80()
  test_flip_actor_obs_involution_unified229()
  test_flip_critic_lin_vel_at_tail()
  print("ok")
