"""Symmetry flip sanity checks — G1 legs+waist; arms kept; grasp ty/arm mirrored."""

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


def test_flip_actions_matches_homie12():
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu")
  x = torch.randn(8, 12)
  y = ppo.flip_actions(ppo.flip_actions(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_keeps_arms_unchanged():
  """Arms are random curriculum — flip must not L↔R-swap or zero them."""
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(4, 480)
  y = ppo.flip_actor_obs(x)
  for t in range(6):
    q0 = t * 80 + 10
    assert torch.allclose(y[:, q0 + 15 : q0 + 29], x[:, q0 + 15 : q0 + 29])
    assert torch.allclose(
      y[:, q0 + 29 + 15 : q0 + 58], x[:, q0 + 29 + 15 : q0 + 58]
    )


def test_flip_actor_obs_involution_legs():
  """Legs + waist + actions round-trip under double flip."""
  ac = HIMActorCritic(480, 83, 80, 83, 6, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(4, 480)
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_mirrors_grasp_ty_and_arm():
  """Unified layout: grasp after a17; mirror ty, swap arm 1↔2; keep flags."""
  d = HOMIE_ONE_STEP_OBS
  ac = HIMActorCritic(d, d + 3, d, d + 3, 1, 1, 14)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.zeros(2, d)
  g0 = HOMIE_PROPRIO_DIM + HOMIE_ACTION_OBS_DIM
  x[:, g0 + 0] = 0.0  # is_target
  x[:, g0 + 1] = 1.0  # is_waist
  x[:, g0 + 2] = 0.15  # extend_dist
  x[:, g0 + 3] = 0.3  # tx
  x[:, g0 + 4] = 0.2  # ty
  x[:, g0 + 5] = -0.1  # tz
  x[:, g0 + 6] = 1.0  # arm left
  y = ppo.flip_actor_obs(x)
  assert torch.allclose(y[:, g0 + 1], x[:, g0 + 1])
  assert torch.allclose(y[:, g0 + 4], -x[:, g0 + 4])
  assert torch.allclose(y[:, g0 + 6], torch.full_like(x[:, g0 + 6], 2.0))
  z = ppo.flip_actor_obs(y)
  assert torch.allclose(z[:, g0 + 4], x[:, g0 + 4])
  assert torch.allclose(z[:, g0 + 6], x[:, g0 + 6])


def test_flip_does_not_zero_sdf_tail():
  """SDF/clearance/EE trail stay as-is (zeros in p1/p2; live in p3)."""
  d = HOMIE_ONE_STEP_OBS
  ac = HIMActorCritic(d, d + 3, d, d + 3, 1, 1, 12)
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  x = torch.randn(2, d)
  y = ppo.flip_actor_obs(x)
  tail0 = HOMIE_PROPRIO_DIM + HOMIE_ACTION_OBS_DIM + HOMIE_GRASP_DIM
  assert torch.allclose(y[:, tail0:], x[:, tail0:])


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


if __name__ == "__main__":
  test_unified_layout_constants()
  test_flip_actions_matches_homie12()
  test_flip_keeps_arms_unchanged()
  test_flip_actor_obs_involution_legs()
  test_flip_mirrors_grasp_ty_and_arm()
  test_flip_does_not_zero_sdf_tail()
  test_flip_critic_lin_vel_at_tail()
  print("ok")
