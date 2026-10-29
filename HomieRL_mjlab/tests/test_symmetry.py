"""Symmetry flip sanity checks for phase-1/2/3 obs layouts."""

import torch

from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic
from homierl_mjlab.rl.him.him_ppo import HIMPPO


def _ppo(one_step: int, num_actions: int, critic_one: int | None = None) -> HIMPPO:
  c = one_step if critic_one is None else critic_one
  ac = HIMActorCritic(
    one_step * 6, c * 1, one_step, c, 6, 1, num_actions
  )
  ppo = HIMPPO(ac, use_flip=True, device="cpu", num_dof=29)
  ppo._joint_perm = list(range(29))
  ppo._joint_sign = [1.0] * 29
  return ppo


def test_flip_actions_involution():
  for n_act in (12, 14, 17):
    ppo = _ppo(80, n_act)
    x = torch.randn(8, n_act)
    y = ppo.flip_actions(ppo.flip_actions(x))
    assert torch.allclose(x, y, atol=1e-5), n_act


def test_flip_actor_obs_involution_phase1():
  # proprio68 + a12 + ith1 + tz1 + container10 = 92
  ppo = _ppo(92, 12)
  x = torch.randn(4, 92 * 6)
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_phase2_grasp_and_is_waist_idx():
  # proprio68 + a14 + grasp7 + container10 = 99; is_waist @ 83
  one = 99
  ppo = _ppo(one, 14, critic_one=one + 3)
  assert ppo.is_waist_one_step_idx == 83  # default before set_teacher
  step = torch.zeros(2, one)
  step[:, 83] = 1.0  # is_waist
  step[:, 82 + 4] = 0.3  # ty (g0=82 → ty at 86)
  step[:, 82 + 6] = 1.0  # arm=left
  flipped = ppo._flip_one_step(step, with_lin_vel=False)
  assert torch.allclose(flipped[:, 83], step[:, 83])  # is_waist unchanged
  assert torch.allclose(flipped[:, 86], -step[:, 86])  # ty mirrored
  assert torch.allclose(flipped[:, 88], torch.full((2,), 2.0))  # arm L→R
  # round-trip (arm labels must be discrete 0/1/2 — swap is not soft)
  x = torch.randn(4, one * 6)
  for t in range(6):
    x[:, t * one + 88] = torch.tensor([0.0, 1.0, 2.0, 1.0])
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-5)


def test_flip_critic_lin_vel_is_trailing():
  # Phase-2 critic: 99 + lin_vel3 = 102. Must NOT treat idx 82 as lin_vel.
  one = 99
  ppo = _ppo(one, 14, critic_one=one + 3)
  step = torch.zeros(2, one + 3)
  step[:, one + 1] = 0.5  # lin_vel y
  step[:, 83] = 1.0  # is_waist must stay put under lin_vel flip
  flipped = ppo._flip_one_step(step, with_lin_vel=True)
  assert torch.allclose(flipped[:, 83], step[:, 83])
  assert torch.allclose(flipped[:, one + 1], -step[:, one + 1])


def test_flip_phase3_rich_trail():
  # 68+17+7+128+6+3 = 229; is_waist @ 86
  one = 229
  ppo = _ppo(one, 17, critic_one=one + 3)
  step = torch.randn(2, one)
  step[:, 86] = 1.0
  flipped = ppo._flip_one_step(step, with_lin_vel=False)
  assert torch.allclose(flipped[:, 86], step[:, 86])
  # EE err y at 68+17+7+128+6+1 = 227
  assert torch.allclose(flipped[:, 227], -step[:, 227])
  x = torch.randn(2, one * 6)
  for t in range(6):
    x[:, t * one + 86 + 5] = torch.tensor([1.0, 2.0])  # arm @ g0+6, g0=85
  y = ppo.flip_actor_obs(ppo.flip_actor_obs(x))
  assert torch.allclose(x, y, atol=1e-4)


def test_loco_mask_phase2_uses_is_waist():
  one = 99
  ppo = _ppo(one, 14)
  ppo.is_waist_one_step_idx = 83
  ppo.keep_loco_always = False
  h = 6
  obs = torch.zeros(4, h * one)
  # last frame is_waist: envs 0,1 waist-on → loco mask 0
  obs[0, (h - 1) * one + 83] = 1.0
  obs[1, (h - 1) * one + 83] = 1.0
  mask = ppo._loco_mask(obs)
  assert torch.allclose(mask, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_loco_mask_phase3_always():
  ppo = _ppo(229, 17)
  ppo.keep_loco_always = True
  obs = torch.randn(3, 6 * 229)
  mask = ppo._loco_mask(obs)
  assert torch.allclose(mask, torch.ones(3))


if __name__ == "__main__":
  test_flip_actions_involution()
  test_flip_actor_obs_involution_phase1()
  test_flip_phase2_grasp_and_is_waist_idx()
  test_flip_critic_lin_vel_is_trailing()
  test_flip_phase3_rich_trail()
  test_loco_mask_phase2_uses_is_waist()
  test_loco_mask_phase3_always()
  print("ok")
