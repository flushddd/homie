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


def _episode_mean_reward(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  default_weight: float = 1.5,
) -> tuple[float, float] | None:
  """Mean episode reward + term weight for ``env_ids``, or None if unavailable.

  mjlab stores Σ(func * weight * dt); divide by episode_length_s → mean(func)*weight.
  Must be called from a reset event *before* ``reward_manager.reset``.
  """
  if env_ids is None or len(env_ids) == 0:
    return None
  max_len = int(env.max_episode_length)
  if max_len <= 0 or int(env.common_step_counter) % max_len != 0:
    return None
  rm = getattr(env, "reward_manager", None)
  if rm is None or not hasattr(rm, "_episode_sums"):
    return None
  sums = rm._episode_sums.get(reward_name)
  if sums is None:
    return None
  weight = float(default_weight)
  term = rm.cfg.get(reward_name) if isinstance(rm.cfg, dict) else None
  if term is not None:
    weight = float(getattr(term, "weight", weight))
  ep_len_s = float(env.max_episode_length_s)
  if ep_len_s <= 0.0:
    return None
  return torch.mean(sums[env_ids]).item() / ep_len_s, weight


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
  if state is None:
    return
  got = _episode_mean_reward(env, env_ids, reward_name)
  if got is None:
    return
  mean_ep, weight = got
  if mean_ep > threshold_frac * weight:
    state.ratio = min(1.0, state.ratio + float(ratio_step))
    # Mirror Homie extras["episode"]["action_curriculum_ratio"].
    log = env.extras.setdefault("log", {})
    log["Curriculum/arm_action_ratio"] = torch.tensor(state.ratio, device=env.device)


def bump_reach_mode_curriculum_from_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str = "tracking_x_vel",
  threshold_frac: float = 0.8,
  ratio_step: float = 0.05,
) -> None:
  """Phase-3: bump reach-mode alpha when loco tracking clears the threshold.

  Same gate as upper-arm random curriculum (``tracking_x_vel`` ≥
  ``threshold_frac * weight``). Raises ``grasp.reach_ratio`` 0→1; mode mix
  lerps ``mode_probs_start`` → ``mode_probs`` with that ratio (reach share
  opens only after loco is stable).
  """
  try:
    grasp = env.command_manager.get_term("grasp")
  except Exception:  # noqa: BLE001
    return
  if grasp is None or not bool(getattr(grasp.cfg, "enable_arm_reach", False)):
    return
  got = _episode_mean_reward(env, env_ids, reward_name)
  if got is None:
    return
  mean_ep, weight = got
  if mean_ep > threshold_frac * weight:
    grasp.reach_ratio = min(1.0, float(grasp.reach_ratio) + float(ratio_step))
    log = env.extras.setdefault("log", {})
    log["Curriculum/reach_mode_ratio"] = torch.tensor(
      grasp.reach_ratio, device=env.device
    )


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


def log_waist_pitch(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  env_index: int = 0,
  every_steps: int = 25,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", joint_names=("waist_pitch_joint",)
  ),
) -> None:
  """Print waist pitch (rad/deg) + policy action for one env during play."""
  del env_ids
  step = int(getattr(env, "common_step_counter", 0))
  if every_steps > 1 and (step % int(every_steps)) != 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  jids = torch.as_tensor(
    asset_cfg.joint_ids, device=env.device, dtype=torch.long
  ).view(-1)
  if jids.numel() == 0:
    return
  idx = min(max(int(env_index), 0), env.num_envs - 1)
  q = float(asset.data.joint_pos[idx, jids[0]].item())
  # Phase-2/3 loco actions: last of the 14 joint_pos block is waist_pitch.
  act_pitch = float("nan")
  try:
    term = env.action_manager.get_term("joint_pos")
    names = list(getattr(term, "target_names", []) or [])
    if "waist_pitch_joint" in names:
      ai = names.index("waist_pitch_joint")
      act_pitch = float(env.action_manager.action[idx, ai].item())
    elif env.action_manager.total_action_dim >= 14:
      act_pitch = float(env.action_manager.action[idx, 13].item())
  except Exception:  # noqa: BLE001
    pass
  print(
    f"[play] step={step} env={idx} "
    f"waist_pitch={q:.4f} rad ({q * 57.2957795:.2f} deg) "
    f"act_pitch={act_pitch:.4f}",
    flush=True,
  )
