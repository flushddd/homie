"""Symmetry flip sanity checks — aligned with Homie ``flip_g1_*``."""

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


def test_layout_constants():
  # Phase-3 full width constant; phase-2 is_waist @ 68+14+1.
  assert HOMIE_ONE_STEP_OBS == (
    HOMIE_PROPRIO_DIM
    + HOMIE_ACTION_OBS_DIM
    + HOMIE_GRASP_DIM
    + HOMIE_SDF_DIM
    + HOMIE_CLEARANCE_DIM
    + HOMIE_EE_ERR_DIM
  )
  assert HOMIE_ONE_STEP_OBS == 229
  assert HOMIE_IS_WAIST_IDX == 83


def test_flip_actions_matches_homie12():
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu")
  x = torch.randn(8, 12)
  y = ppo.flip_actions(ppo.flip_actions(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_zeros_arms_like_homie():
  """Homie leaves arm q/dq at 0 in flipped obs (arm flip commented out)."""
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(4, 480)
  y = ppo.flip_actor_obs(x)
  for t in range(6):
    q0 = t * 80 + 10
    assert torch.allclose(y[:, q0 + 15 : q0 + 29], torch.zeros(4, 14), atol=1e-7)
    assert torch.allclose(
      y[:, q0 + 29 + 15 : q0 + 58], torch.zeros(4, 14), atol=1e-7
    )


def test_flip_actor_obs_involution_legs():
  """Legs + waist + actions round-trip; arms/grasp stay 0 (Homie zeros_like)."""
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(4, 480)
  for t in range(6):
    base = t * 80
    q0 = base + 10
    x[:, q0 + 15 : q0 + 29] = 0
    x[:, q0 + 29 + 15 : q0 + 58] = 0
    if 80 > 68 + 12:
      x[:, base + 68 + 12 : base + 80] = 0
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_zeros_grasp_sdf_tail():
  """Homie zeros_like → grasp/SDF stay 0 in flipped obs even if present."""
  d = HOMIE_ONE_STEP_OBS
  ac = HIMActorCritic(d, d + 3, d, d + 3, 1, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(2, d)
  y = ppo.flip_actor_obs(x)
  # after proprio68 + a12 (we write n_act=12), rest of 17-pad + grasp/SDF = 0
  assert torch.allclose(y[:, 68 + 12 :], torch.zeros_like(y[:, 68 + 12 :]))


def test_flip_critic_lin_vel_at_tail():
  d = HOMIE_ONE_STEP_OBS
  dc = d + 3
  ac = HIMActorCritic(d, dc, d, dc, 1, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.zeros(2, dc)
  x[:, -3:] = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
  y = ppo.flip_critic_obs(x)
  assert torch.allclose(y[:, -3], x[:, -3])
  assert torch.allclose(y[:, -2], -x[:, -2])
  assert torch.allclose(y[:, -1], x[:, -1])


def test_phase1_critic_lin_vel_last_not_container():
  """p1 critic = actor90 (…|container10) + lin_vel3; flip must hit lin_vel."""
  d_actor, d_crit = 90, 93
  ac = HIMActorCritic(d_actor * 6, d_crit, d_actor, d_crit, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.zeros(2, d_crit)
  # container pad (zeros) then lin_vel at tail
  x[:, -3:] = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.25]])
  y = ppo.flip_critic_obs(x)
  assert torch.allclose(y[:, -3], x[:, -3])
  assert torch.allclose(y[:, -2], -x[:, -2])
  assert torch.allclose(y[:, -1], x[:, -1])
  # if someone put container after lin_vel, -3 would be wrong — guard length
  assert d_crit == d_actor + 3


if __name__ == "__main__":
  test_layout_constants()
  test_flip_actions_matches_homie12()
  test_flip_zeros_arms_like_homie()
  test_flip_actor_obs_involution_legs()
  test_flip_zeros_grasp_sdf_tail()
  test_flip_critic_lin_vel_at_tail()
  test_phase1_critic_lin_vel_last_not_container()
  print("ok")
