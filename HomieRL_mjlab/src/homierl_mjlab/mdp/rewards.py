"""HOMIE paper G1 rewards, aligned with HomieRL ``G1RoughCfg`` / ``legged_robot``.

Intentionally excludes loco-manipulation extras from the evolved inspire-waist
stack (reach_*, waist_*, target_height masks). Those belong to a later phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command_height_target(
  env: ManagerBasedRlEnv,
  command_name: str,
  base_height_target: float,
) -> torch.Tensor:
  """Absolute base-height target from twist+height command channel 3."""
  cmd = env.command_manager.get_command(command_name)
  return base_height_target + cmd[:, 3]


def _foot_site_ids(asset: Entity) -> list[int]:
  return list(asset.find_sites(("left_foot", "right_foot"))[0])


def _base_height_from_feet(
  asset: Entity,
  ankle_sole_distance: float = 0.02,
) -> torch.Tensor:
  """Homie style: max(root_z - foot_z) + sole offset."""
  root_z = asset.data.root_link_pos_w[:, 2]
  foot_ids = _foot_site_ids(asset)
  feet_z = asset.data.site_pos_w[:, foot_ids, 2]
  return torch.max(root_z.unsqueeze(-1) - feet_z, dim=-1).values + ankle_sole_distance


# ---------------------------------------------------------------------------
# Velocity / posture tracking (paper)
# ---------------------------------------------------------------------------


def tracking_x_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  tracking_sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  err = torch.square(cmd[:, 0] - asset.data.root_link_lin_vel_b[:, 0])
  return torch.exp(-err / tracking_sigma)


def tracking_y_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  tracking_sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  err = torch.square(cmd[:, 1] - asset.data.root_link_lin_vel_b[:, 1])
  return torch.exp(-err / tracking_sigma)


def tracking_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  tracking_sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  err = torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / tracking_sigma)


def lin_vel_z(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize vertical base velocity only near standing height (Homie gate)."""
  asset: Entity = env.scene[asset_cfg.name]
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()
  return torch.square(asset.data.root_link_lin_vel_b[:, 2]) * gate


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize non-flat base via projected gravity xy (Homie ``orientation``)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def tracking_base_height(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  ankle_sole_distance: float = 0.02,
  height_error_scale: float = 4.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Homie ``r_height``: ``exp(-|h - h*| * 4)``."""
  asset: Entity = env.scene[asset_cfg.name]
  h_target = _command_height_target(env, command_name, base_height_target)
  try:
    base_height = _base_height_from_feet(asset, ankle_sole_distance)
  except Exception:
    base_height = asset.data.root_link_pos_w[:, 2]
  height_error = torch.abs(base_height - h_target)
  return torch.exp(-height_error * height_error_scale)


# ---------------------------------------------------------------------------
# Joint regularization
# ---------------------------------------------------------------------------


def deviation_hip_joint(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"),
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()
  err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
    :, asset_cfg.joint_ids
  ]
  return torch.sum(torch.square(err), dim=-1) * gate


def deviation_ankle_joint(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", joint_names=(".*_ankle_roll_joint",)
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()
  err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[
    :, asset_cfg.joint_ids
  ]
  return torch.sum(torch.square(err), dim=-1) * gate


def deviation_knee_joint(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", joint_names=(".*_knee_joint",)
  ),
) -> torch.Tensor:
  """Homie ``r_knee``: |(q_norm - 0.5) * height_error| summed over knees."""
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  # Homie: height_error = root_states[:,2] - h_target
  height_error = asset.data.root_link_pos_w[:, 2] - h_target

  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  # Soft limits as action range proxy (Homie used action_min/max * scale).
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # [N,J,2]
  q_min = limits[..., 0]
  q_max = limits[..., 1]
  joint_deviation = (q - q_min) / (q_max - q_min + 1e-6)
  return torch.sum(
    torch.abs((joint_deviation - 0.5) * height_error.unsqueeze(-1)), dim=-1
  )


def dof_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc), dim=-1)


def dof_vel_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=-1)


def dof_vel_limits(
  env: ManagerBasedRlEnv,
  soft_ratio: float = 0.80,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  # MuJoCo joint velocity limits if present; else skip soft gate via large bound.
  limits = getattr(asset.data, "joint_vel_limits", None)
  if limits is None:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.sum(
    (torch.abs(asset.data.joint_vel) - limits * soft_ratio).clip(min=0.0), dim=-1
  )


def torques_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.actuator_force), dim=-1)


def torque_limits(
  env: ManagerBasedRlEnv,
  soft_ratio: float = 0.95,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Soft torque-limit penalty; inactive if actuator force limits are unavailable."""
  del soft_ratio, asset_cfg
  # mjlab does not yet expose per-actuator effort limits on EntityData.
  return torch.zeros(env.num_envs, device=env.device)


def joint_power(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  # Match Homie: |qdot| * |tau|, normalized by commanded xy/yaw energy.
  n_act = asset.data.actuator_force.shape[-1]
  vel = asset.data.joint_vel[:, :n_act]
  power = torch.sum(torch.abs(vel) * torch.abs(asset.data.actuator_force), dim=-1)
  denom = torch.clip(
    torch.sum(torch.square(cmd[:, 0:2]), dim=-1) + 0.2 * torch.square(cmd[:, 2]),
    min=0.1,
  )
  return power / denom


def joint_tracking_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Homie ``joint_tracking_error``: ||q_target - q||^2 (legs + upper curriculum)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.joint_pos_target - asset.data.joint_pos), dim=-1
  )


def action_vanish(
  env: ManagerBasedRlEnv,
  clip: float = 1.0,
) -> torch.Tensor:
  """Penalize raw actions outside [-clip, clip] (Homie action_vanish)."""
  actions = env.action_manager.action
  upper = torch.clip(actions - clip, min=0.0)
  lower = torch.clip(-clip - actions, min=0.0)
  return torch.sum(upper + lower, dim=-1)


def smoothness_second_order(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Second-order action smoothness (Homie ``smoothness``)."""
  am = env.action_manager
  return torch.sum(
    torch.square(am.action - 2.0 * am.prev_action + am.prev_prev_action), dim=-1
  )


# ---------------------------------------------------------------------------
# Feet / contact
# ---------------------------------------------------------------------------


def _contact_forces(
  env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  assert sensor.data.force is not None
  return sensor.data.force  # [N, F, 3]


def _in_contact(
  env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact", thr: float = 1.0
) -> torch.Tensor:
  forces = _contact_forces(env, sensor_name)
  return torch.norm(forces, dim=-1) > thr  # [N, F]


def feet_air_time_homie(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  threshold: float = 0.5,
) -> torch.Tensor:
  """Homie air-time: (air_time - 0.5) * first_contact, gated by non-zero cmd."""
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  assert sensor.data.last_air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  rew = torch.sum(
    (sensor.data.last_air_time - threshold) * first_contact.float(), dim=-1
  )
  cmd = env.command_manager.get_command(command_name)
  rew = rew * (torch.norm(cmd[:, :3], dim=-1) > 0.1).float()
  return rew


def feet_clearance(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  clearance_height_target: float = 0.14,
  stand_gate: float = 0.71,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  """Penalize foot height error while swinging (Homie feet_clearance)."""
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_gate).float()

  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  # Approximate clearance vs root; Homie used terrain-relative foot height.
  root_z = asset.data.root_link_pos_w[:, 2].unsqueeze(-1)
  # Use foot height above a nominal ground estimate (root - standing height).
  ground = root_z - base_height_target
  feet_height = foot_z - ground
  height_error = torch.square(feet_height - clearance_height_target)

  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  lateral_speed = torch.linalg.norm(foot_vel_xy, dim=-1)
  return torch.sum(height_error * lateral_speed, dim=-1) * gate


def feet_distance_lateral(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  least: float = 0.2,
  most: float = 0.35,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()

  root_pos = asset.data.root_link_pos_w[:, 0:3]
  root_quat = asset.data.root_link_quat_w
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  rel = feet - root_pos.unsqueeze(1)
  # body frame
  n = env.num_envs
  body = torch.zeros(n, 2, 3, device=env.device)
  for i in range(2):
    body[:, i] = quat_apply_inverse(root_quat, rel[:, i])
  lateral = torch.abs(body[:, 0, 1] - body[:, 1, 1])
  # Homie returns clamp terms (negative when outside band) — used with +weight.
  return (
    torch.clamp(lateral - least, max=0.0) + torch.clamp(-lateral + most, max=0.0)
  ) * gate


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  least: float = 0.2,
  most: float = 0.35,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot",
    body_names=(
      "left_knee_link",
      "left_hip_yaw_link",
      "right_knee_link",
      "right_hip_yaw_link",
    ),
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()

  root_pos = asset.data.root_link_pos_w[:, 0:3]
  root_quat = asset.data.root_link_quat_w
  knees = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  rel = knees - root_pos.unsqueeze(1)
  n, k, _ = rel.shape
  body = torch.zeros(n, k, 3, device=env.device)
  for i in range(k):
    body[:, i] = quat_apply_inverse(root_quat, rel[:, i])
  # Homie: |L_knee.y - R_hip_yaw.y| + |L_hip_yaw.y - R_knee.y| with 4 bodies.
  if k >= 4:
    lateral = torch.abs(body[:, 0, 1] - body[:, 2, 1]) + torch.abs(
      body[:, 1, 1] - body[:, 3, 1]
    )
  else:
    lateral = torch.abs(body[:, 0, 1] - body[:, 1, 1])
  return (
    torch.clamp(lateral - least * 2, max=0.0)
    + torch.clamp(-lateral + most * 2, max=0.0)
  ) * gate


def feet_parallel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  base_height_target: float = 0.74,
  stand_height_threshold: float = 0.735,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  """Variance of multi-point foot distances — approx with site pair variance proxy."""
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  # Single pair: use squared height difference as parallel proxy.
  dz = feet[:, 0, 2] - feet[:, 1, 2]
  return torch.square(dz) * gate


def feet_ground_parallel(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  """Penalize foot height variance while in sustained contact."""
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  sensor: ContactSensor = env.scene.sensors[sensor_name]
  contact = (sensor.data.found > 0).float() if sensor.data.found is not None else _in_contact(env, sensor_name).float()
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  # Per-env variance across feet, gated by both-contact.
  mean_z = feet_z.mean(dim=-1, keepdim=True)
  var = torch.mean(torch.square(feet_z - mean_z), dim=-1)
  both = (contact.sum(dim=-1) >= 2).float()
  return var * both


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  forces = _contact_forces(env, sensor_name)
  lateral = torch.norm(forces[:, :, :2], dim=-1)
  vertical = torch.abs(forces[:, :, 2])
  return torch.any(lateral > 3.0 * vertical, dim=-1).float()


def feet_slip_homie(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  contact = _in_contact(env, sensor_name, thr=1.0).float()
  vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return torch.sum(torch.linalg.norm(vel_xy, dim=-1) * contact, dim=-1)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  max_contact_force: float = 400.0,
) -> torch.Tensor:
  forces = _contact_forces(env, sensor_name)
  mag = torch.linalg.norm(forces, dim=-1)
  return torch.sum((mag - max_contact_force).clip(min=0.0), dim=-1)


def contact_momentum(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot")
  ),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  asset_cfg.resolve(env.scene)
  forces = _contact_forces(env, sensor_name)
  vz = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
  mom = torch.clip(vz, max=0.0) * torch.clip(forces[:, :, 2] - 50.0, min=0.0)
  return torch.sum(mom, dim=-1)


def no_fly(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
) -> torch.Tensor:
  contact = _in_contact(env, sensor_name, thr=0.5)
  single = (contact.float().sum(dim=-1) == 1).float()
  cmd = env.command_manager.get_command(command_name)
  standing = (torch.norm(cmd[:, :3], dim=-1) < 0.1).float()
  return torch.maximum(single, standing)


def stand_still(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "twist",
  base_height_target: float = 0.74,
  stand_height_threshold: float = 0.735,
) -> torch.Tensor:
  forces = _contact_forces(env, sensor_name)
  # Homie: count feet with low vertical force as "not firmly planted".
  light = (forces[:, :, 2] < 0.1).float().sum(dim=-1)
  h_target = _command_height_target(env, command_name, base_height_target)
  gate = (h_target >= stand_height_threshold).float()
  cmd = env.command_manager.get_command(command_name)
  zero_cmd = (torch.norm(cmd[:, :3], dim=-1) < 0.1).float()
  return light * gate * zero_cmd
