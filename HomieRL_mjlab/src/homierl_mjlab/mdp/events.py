"""Upper-body pose curriculum and related events (paper HOMIE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ENV_STATE_KEY = "_homie_upper_curriculum"


class UpperBodyCurriculumState:
  """Per-env buffers for smoothly interpolated upper-body targets."""

  def __init__(
    self,
    num_envs: int,
    num_joints: int,
    device: torch.device | str,
    init_ratio: float = 0.0,
  ):
    self.ratio = float(init_ratio)
    self.current = torch.zeros(num_envs, num_joints, device=device)
    self.target = torch.zeros(num_envs, num_joints, device=device)
    self.delta = torch.zeros(num_envs, num_joints, device=device)
    self.joint_ids: torch.Tensor | None = None
    self.joint_pos_limits: torch.Tensor | None = None  # (J, 2)


def _get_state(env: ManagerBasedRlEnv) -> UpperBodyCurriculumState | None:
  return getattr(env, _ENV_STATE_KEY, None)


def init_upper_body_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_shoulder_.*",
      ".*_elbow_joint",
      ".*_wrist_.*",
    ),
  ),
  init_ratio: float = 0.0,
) -> None:
  """Startup: resolve upper joints and allocate curriculum buffers."""
  del env_ids
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  joint_ids = list(asset_cfg.joint_ids)
  num_j = len(joint_ids)
  state = UpperBodyCurriculumState(
    num_envs=env.num_envs,
    num_joints=num_j,
    device=env.device,
    init_ratio=init_ratio,
  )
  state.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  # soft joint limits (N, J, 2) -> take first env
  limits = asset.data.soft_joint_pos_limits[0, joint_ids].clone()
  state.joint_pos_limits = limits
  # start at default pose
  default = asset.data.default_joint_pos[:, joint_ids].clone()
  state.current[:] = default
  state.target[:] = default
  setattr(env, _ENV_STATE_KEY, state)


def bump_upper_body_curriculum_from_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str = "tracking_x_vel",
  threshold_frac: float = 0.8,
  ratio_step: float = 0.05,
) -> None:
  """Homie ``update_action_curriculum`` (legged_robot1jieduan).

  Only when ``common_step_counter % max_episode_length == 0`` and mean episode
  ``tracking_x_vel`` exceeds ``threshold_frac * weight``, bump ratio by
  ``ratio_step``. Must run in a reset event *before* ``reward_manager.reset``.
  """
  state = _get_state(env)
  if state is None or env_ids is None or len(env_ids) == 0:
    return
  max_len = int(env.max_episode_length)
  if max_len <= 0 or int(env.common_step_counter) % max_len != 0:
    return
  rm = getattr(env, "reward_manager", None)
  if rm is None or not hasattr(rm, "_episode_sums"):
    return
  sums = rm._episode_sums.get(reward_name)
  if sums is None:
    return
  # Resolve weight from reward cfg (Homie compares to ``reward_scales[name]``).
  weight = 1.5
  term = rm.cfg.get(reward_name) if isinstance(rm.cfg, dict) else None
  if term is not None:
    weight = float(getattr(term, "weight", weight))
  # mjlab stores Σ(func * weight * dt); / episode_length_s → mean(func)*weight.
  ep_len_s = float(env.max_episode_length_s)
  if ep_len_s <= 0.0:
    return
  mean_ep = torch.mean(sums[env_ids]).item() / ep_len_s
  if mean_ep > threshold_frac * weight:
    state.ratio = min(1.0, state.ratio + float(ratio_step))
    # Mirror Homie extras["episode"]["action_curriculum_ratio"].
    log = env.extras.setdefault("log", {})
    log["Curriculum/arm_action_ratio"] = torch.tensor(state.ratio, device=env.device)


def update_upper_body_pose_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_shoulder_.*",
      ".*_elbow_joint",
      ".*_wrist_.*",
    ),
  ),
  upper_interval: int = 50,
  respect_play_arm_gate: bool = False,
) -> None:
  """Homie ``random_upper_actions`` (legged_robot1jieduan when ``not arm_on``).

  Every ``upper_interval`` steps sample a new arm target toward a random joint
  limit with exponential bias controlled by ``ratio``; each step interpolates
  ``current → target``. Ratio is bumped separately via tracking
  (``bump_upper_body_curriculum_from_tracking``), matching Homie.

  When ``respect_play_arm_gate`` is True (play), motion runs only if
  ``env._homie_play_arm_random`` is set by the Viser checkbox; otherwise arms
  are held at default.
  """
  del env_ids
  state = _get_state(env)
  if state is None:
    init_upper_body_curriculum(env, None, asset_cfg=asset_cfg)
    state = _get_state(env)
  assert state is not None and state.joint_ids is not None
  assert state.joint_pos_limits is not None

  asset: Entity = env.scene[asset_cfg.name]

  if respect_play_arm_gate and not bool(
    getattr(env, "_homie_play_arm_random", False)
  ):
    default = asset.data.default_joint_pos[:, state.joint_ids]
    state.current[:] = default
    state.target[:] = default
    state.delta.zero_()
    asset.set_joint_position_target(state.current, joint_ids=state.joint_ids)
    return

  step = int(env.common_step_counter)

  # Homie: resample when ``common_step_counter % upper_interval == 0``.
  if step % upper_interval == 0:
    n = env.num_envs
    j = state.current.shape[1]
    # Homie::
    #   r = min(action_curriculum_ratio, 1)
    #   uu ~ U(0,1)
    #   random_upper_ratio = -1/(20*(1-r*0.99)) * log(1-uu+uu*exp(-20*(1-r*0.99)))
    #   random_joint_ratio = random_upper_ratio * U(0,1)
    #   action = (action_min|action_max) * random_joint_ratio
    #   q = default + action_scale * action  ≡  default + (limit-default)*ratio
    r = min(state.ratio, 1.0)
    denom = 20.0 * (1.0 - r * 0.99) + 1e-6
    uu = torch.rand(n, j, device=env.device)
    rand_ratio = (
      -1.0
      / denom
      * torch.log(1.0 - uu + uu * torch.exp(torch.tensor(-denom, device=env.device)))
    )
    rand_ratio = rand_ratio * torch.rand(n, j, device=env.device)

    lo = state.joint_pos_limits[:, 0].unsqueeze(0)
    hi = state.joint_pos_limits[:, 1].unsqueeze(0)
    default = asset.data.default_joint_pos[:, state.joint_ids]
    # Homie: rand_pos>=0 → action_min (lo); else action_max (hi).
    side = torch.rand(n, j, device=env.device) - 0.5
    extreme = torch.where(side >= 0.0, lo, hi)
    state.target = default + (extreme - default) * rand_ratio
    state.delta = (state.target - state.current) / float(upper_interval)

  state.current = state.current + state.delta
  asset.set_joint_position_target(state.current, joint_ids=state.joint_ids)


def reset_upper_body_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=(
      ".*_shoulder_.*",
      ".*_elbow_joint",
      ".*_wrist_.*",
    ),
  ),
  reward_name: str = "tracking_x_vel",
  threshold_frac: float = 0.8,
  ratio_step: float = 0.05,
) -> None:
  """On reset: Homie action-curriculum bump (if due), then snap arm pose."""
  # Episode sums still valid here (event reset runs before reward_manager.reset).
  bump_upper_body_curriculum_from_tracking(
    env,
    env_ids,
    reward_name=reward_name,
    threshold_frac=threshold_frac,
    ratio_step=ratio_step,
  )
  state = _get_state(env)
  if state is None or state.joint_ids is None:
    return
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos[env_ids][:, state.joint_ids]
  state.current[env_ids] = default
  state.target[env_ids] = default
  state.delta[env_ids] = 0.0
  asset.set_joint_position_target(state.current[env_ids], joint_ids=state.joint_ids, env_ids=env_ids)


def get_curriculum_ratio(env: ManagerBasedRlEnv) -> float:
  state = _get_state(env)
  return 0.0 if state is None else state.ratio


def hold_joints_at_default(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=(
      "waist_.*",
      ".*_shoulder_.*",
      ".*_elbow_joint",
      ".*_wrist_.*",
    ),
  ),
) -> None:
  """Keep listed joints' PD targets at ``default_joint_pos`` (Homie action=0)."""
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  # Must be a 1-D LongTensor so Entity._outer_index can broadcast with env_ids.
  joint_ids = torch.as_tensor(
    asset_cfg.joint_ids, device=env.device, dtype=torch.long
  ).view(-1)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).view(-1)
  default = asset.data.default_joint_pos[env_ids][:, joint_ids]
  asset.set_joint_position_target(default, joint_ids=joint_ids, env_ids=env_ids)
