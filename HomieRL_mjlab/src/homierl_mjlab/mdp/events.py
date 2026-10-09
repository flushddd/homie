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
  ratio_step: float = 0.01,
  warmup_steps: int = 20_000,
  steps_per_ratio_bump: int = 5_000,
) -> None:
  """Interval event: advance curriculum and set upper-body position targets.

  Slower than the original Homie bump schedule so locomotion can stabilize
  before large upper-body disturbances:
  - no ratio increase before ``warmup_steps``
  - then ``ratio += ratio_step`` every ``steps_per_ratio_bump`` env steps
  - sample sparse random poses toward joint limits with exponential bias
  - interpolate toward the new target over ``upper_interval`` control steps
  """
  state = _get_state(env)
  if state is None:
    init_upper_body_curriculum(env, None, asset_cfg=asset_cfg)
    state = _get_state(env)
  assert state is not None and state.joint_ids is not None
  assert state.joint_pos_limits is not None

  asset: Entity = env.scene[asset_cfg.name]
  step = int(env.common_step_counter)

  # Delayed, slow curriculum ramp (prevents mid-training collapse).
  if step >= warmup_steps and steps_per_ratio_bump > 0:
    if (step - warmup_steps) % steps_per_ratio_bump == 0:
      state.ratio = min(1.0, state.ratio + ratio_step)

  # Resample targets periodically.
  if step % upper_interval == 0:
    n = env.num_envs
    j = state.current.shape[1]
    uu = torch.rand(n, j, device=env.device)
    # Inverse-CDF style concentration near 0 when ratio small (paper Homie).
    r = min(state.ratio, 1.0)
    denom = 20.0 * (1.0 - r * 0.99) + 1e-6
    rand_ratio = -1.0 / denom * torch.log(1.0 - uu + uu * torch.exp(torch.tensor(-denom, device=env.device)))
    rand_ratio = rand_ratio * torch.rand(n, j, device=env.device)

    lo = state.joint_pos_limits[:, 0].unsqueeze(0)
    hi = state.joint_pos_limits[:, 1].unsqueeze(0)
    default = asset.data.default_joint_pos[:, state.joint_ids]
    # Blend from default toward a random limit-side pose.
    side = torch.rand(n, j, device=env.device) - 0.5
    extreme = torch.where(side >= 0, hi, lo)
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
) -> None:
  """On reset: snap curriculum pose to default for finished envs."""
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
