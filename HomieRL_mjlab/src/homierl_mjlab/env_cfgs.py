"""HOMIE flat G1 environment configuration for mjlab."""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from homierl_mjlab import mdp as homie_mdp
from homierl_mjlab.robot.g1_constants import (
  ARM_JOINT_NAMES,
  FOOT_COLLISION_GEOM_NAMES,
  HOMIE_ACTION_SCALE,
  HOMIE_LEG_ACTION_SCALE,
  LEG_JOINT_NAMES,
  LOCO_JOINT_NAMES,
  UPPER_JOINT_NAMES,
  WAIST_YAW_JOINT_NAMES,
  get_g1_robot_cfg,
)


def _configure_phase_events(cfg: ManagerBasedRlEnvCfg, phase: int) -> None:
  """Phase-1: PD-hold waist+arms. Phase-2: policy roll/pitch; yaw+arms held."""
  cfg.events.pop("init_upper_curriculum", None)
  cfg.events.pop("upper_body_curriculum", None)
  cfg.events.pop("reset_upper_curriculum", None)
  cfg.events.pop("hold_upper_default", None)
  cfg.events.pop("hold_waist_default", None)
  cfg.events.pop("hold_waist_yaw_default", None)
  cfg.events.pop("hold_arms_default", None)

  if int(phase) <= 1:
    cfg.events["hold_upper_default"] = EventTermCfg(
      func=homie_mdp.hold_joints_at_default,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_JOINT_NAMES),
      },
    )
    return

  # Homie: yaw target forced to init; arms not policy-driven in waist-only stage.
  cfg.events["hold_waist_yaw_default"] = EventTermCfg(
    func=homie_mdp.hold_joints_at_default,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_YAW_JOINT_NAMES),
    },
  )
  cfg.events["hold_arms_default"] = EventTermCfg(
    func=homie_mdp.hold_joints_at_default,
    mode="step",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
    },
  )


def _configure_phase_actions(cfg: ManagerBasedRlEnvCfg, phase: int) -> None:
  """Phase-1: 12 legs. Phase-2: 12 legs + waist roll/pitch (yaw held)."""
  if int(phase) <= 1:
    names = LEG_JOINT_NAMES
    scale = HOMIE_LEG_ACTION_SCALE
  else:
    names = LOCO_JOINT_NAMES
    scale = HOMIE_ACTION_SCALE
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=names,
      scale=scale,
      use_default_offset=True,
    ),
  }


def _configure_phase_commands_and_obs(
  cfg: ManagerBasedRlEnvCfg, phase: int, actor_terms: dict
) -> None:
  """Phase-2 grasp trail; both phases append 10-D container geometry at the end."""
  if int(phase) <= 1:
    cfg.commands.pop("grasp", None)
    actor_terms.pop("grasp_target", None)
  else:
    cfg.commands["grasp"] = homie_mdp.GraspTargetCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 4.0),
      debug_vis=True,
      base_height_target=0.74,
    )
    actor_terms["grasp_target"] = ObservationTermCfg(
      func=homie_mdp.grasp_target_command,
      params={"command_name": "grasp"},
    )

  # Always last in actor/critic obs (after actions / grasp).
  # Fixed 10-D container zeros in phase-1/2 (slot reserved; no signal yet).
  cfg.commands["container"] = homie_mdp.ContainerGeometryCommandCfg(
    entity_name="robot",
    randomize=False,
    fixed_values=(0.0,) * 10,
    debug_vis=False,
  )
  actor_terms["container_geom"] = ObservationTermCfg(
    func=homie_mdp.container_geometry_command,
    params={"command_name": "container"},
  )


def homie_g1_flat_env_cfg(
  play: bool = False, phase: int = 1
) -> ManagerBasedRlEnvCfg:
  """Paper HOMIE G1 flat: phase-1 loco/squat; phase-2 waist + grasp targets."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Flat plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Drop height-scan sensors (flat).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in ("terrain_scan", "foot_height_scan")
  )

  site_names = ("left_foot", "right_foot")
  # Inspire-waist MJCF: 4 named foot spheres per foot after robot patching.
  geom_names = FOOT_COLLISION_GEOM_NAMES

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  # ---- Actions: phase-1 = 12 legs; phase-2 = 12 legs + 3 waist ----
  _configure_phase_actions(cfg, phase=phase)

  # ---- Commands: twist + height (Homie ranges + resampling_time=4s) ----
  # Standing / height / vel bands are applied in TwistHeightCommand._resample_command
  # (Homie set_x schedule). rel_standing_envs unused after that overwrite.
  cfg.commands = {
    "twist": homie_mdp.TwistHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 4.0),
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      heading_command=False,
      debug_vis=True,
      ranges=homie_mdp.TwistHeightCommandCfg.Ranges(
        lin_vel_x=(-0.8, 1.2),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.8, 0.8),
        height=(-0.5, 0.0),
      ),
      base_height_target=0.74,
      viz=homie_mdp.TwistHeightCommandCfg.VizCfg(z_offset=1.15),
    ),
  }

  # ---- Observations (Homie ``compute_observations`` + ``_get_noise_scale_vec``) ----
  # Pipeline: compute → noise → scale. Homie adds noise on already-scaled obs;
  # equivalent here: noise on raw with amplitude = noise_scale / obs_scale inverse
  # arranged as noise_raw = Homie_noise_scale, then scale = Homie obs_scale
  # (see noise_vec = noise_scales * level * obs_scales on scaled channels).
  actor_terms = {
    # Homie: commands get zero noise; twist scaled by (lin_vel, lin_vel, ang_vel).
    "command_twist": ObservationTermCfg(
      func=homie_mdp.twist_command_xy_yaw,
      params={
        "command_name": "twist",
        "scale": (2.0, 2.0, 0.5),
      },
    ),
    "command_height": ObservationTermCfg(
      func=homie_mdp.height_command,
      params={
        "command_name": "twist",
        "absolute": True,
        "base_height_target": 0.74,
      },
    ),
    # Homie: noise_scales.ang_vel=0.5, obs_scales.ang_vel=0.5
    # → noise on raw ±0.5, then *0.5 (matches ±0.25 on scaled obs).
    "base_ang_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      scale=0.5,
    ),
    # Homie: noise_scales.gravity=0.05 (no obs scale).
    "projected_gravity": ObservationTermCfg(
      func=vel_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Homie: noise_scales.dof_pos=0.02, obs_scales.dof_pos=1.0
    "joint_pos": ObservationTermCfg(
      func=vel_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      scale=1.0,
    ),
    # Homie: noise_scales.dof_vel=2.0, obs_scales.dof_vel=0.05
    # → noise on raw ±2.0, then *0.05 (matches ±0.1 on scaled obs).
    "joint_vel": ObservationTermCfg(
      func=vel_mdp.joint_vel_rel,
      noise=Unoise(n_min=-2.0, n_max=2.0),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=vel_mdp.last_action),
  }
  # Grasp (phase-2) then container (10-D) after actions; critic adds lin_vel last.
  _configure_phase_commands_and_obs(cfg, phase, actor_terms)
  critic_terms = {
    **actor_terms,
    "joint_pos": ObservationTermCfg(func=vel_mdp.joint_pos_rel, scale=1.0),
    # Homie critic appends base_lin_vel * obs_scales.lin_vel (2.0).
    # Keep after proprio; container stays at actor tail via shared actor_terms.
    "base_lin_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      scale=2.0,
    ),
  }
  # Re-append container so it remains the final critic term after lin_vel.
  if "container_geom" in critic_terms:
    c_term = critic_terms.pop("container_geom")
    critic_terms["container_geom"] = c_term
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ---- Events (Homie domain_rand + push) ----
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # Homie friction_range = [0.1, 3.0]
  cfg.events["foot_friction"].params["ranges"] = (0.1, 3.0)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  # Homie body_displacement_range ≈ ±0.1 on torso COM.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.1, 0.1),
    1: (-0.1, 0.1),
    2: (-0.1, 0.1),
  }

  # Homie ``_push_robots``: only lin vel xy ±0.5, interval 4s.
  cfg.events["push_robot"] = EventTermCfg(
    func=vel_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(4.0, 4.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
      },
    },
  )

  # Homie initial_joint_pos_scale [0.8, 1.2] + offset [-0.1, 0.1] around default.
  # Approximate with offset range; scale randomization is not a first-class mjlab API.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

  _configure_phase_events(cfg, phase=phase)

  # ---- Rewards: Homie inspire-waist scales (reach_* deferred; waist via is_height) ----
  cfg.rewards = {
    "tracking_x_vel": RewardTermCfg(
      func=homie_mdp.tracking_x_vel,
      weight=1.5,
      params={"command_name": "twist", "tracking_sigma": 0.25},
    ),
    "tracking_y_vel": RewardTermCfg(
      func=homie_mdp.tracking_y_vel,
      weight=1.0,
      params={"command_name": "twist", "tracking_sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=homie_mdp.tracking_ang_vel,
      weight=2.0,
      params={"command_name": "twist", "tracking_sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(
      func=homie_mdp.lin_vel_z,
      weight=-0.5,
      params={"command_name": "twist", "base_height_target": 0.74},
    ),
    "ang_vel_xy": RewardTermCfg(func=homie_mdp.ang_vel_xy, weight=-0.025),
    "orientation": RewardTermCfg(func=homie_mdp.orientation, weight=-1.5),
    "action_rate": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.01),
    "tracking_base_height": RewardTermCfg(
      func=homie_mdp.tracking_base_height,
      weight=2.0,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "ankle_sole_distance": 0.02,
        "height_error_scale": 4.0,
      },
    ),
    "deviation_hip_joint": RewardTermCfg(
      func=homie_mdp.deviation_hip_joint,
      weight=-0.2,
      params={"command_name": "twist", "base_height_target": 0.74},
    ),
    "deviation_ankle_joint": RewardTermCfg(
      func=homie_mdp.deviation_ankle_joint,
      weight=-0.5,
      params={"command_name": "twist", "base_height_target": 0.74},
    ),
    "deviation_knee_joint": RewardTermCfg(
      func=homie_mdp.deviation_knee_joint,
      weight=-0.75,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
      },
    ),
    "deviation_roll_joint": RewardTermCfg(
      func=homie_mdp.deviation_roll_joint, weight=1e-5
    ),
    "deviation_waist_joint": RewardTermCfg(
      func=homie_mdp.deviation_waist_joint,
      weight=-4e-3,
      params={"command_name": "twist"},
    ),
    "dof_acc": RewardTermCfg(func=homie_mdp.dof_acc_l2, weight=-2.5e-7),
    "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-2.0),
    "feet_air_time": RewardTermCfg(
      func=homie_mdp.feet_air_time_homie,
      weight=0.05,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "feet_clearance": RewardTermCfg(
      func=homie_mdp.feet_clearance,
      weight=-0.25,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "clearance_height_target": 0.14,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "feet_distance_lateral": RewardTermCfg(
      func=homie_mdp.feet_distance_lateral,
      weight=0.5,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "least": 0.2,
        "most": 0.35,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "knee_distance_lateral": RewardTermCfg(
      func=homie_mdp.knee_distance_lateral,
      weight=1.0,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "least": 0.2,
        "most": 0.35,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=(
            "left_knee_link",
            "left_hip_yaw_link",
            "right_knee_link",
            "right_hip_yaw_link",
          ),
        ),
      },
    ),
    "feet_ground_parallel": RewardTermCfg(
      func=homie_mdp.feet_ground_parallel,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "feet_parallel": RewardTermCfg(
      func=homie_mdp.feet_parallel,
      weight=-3.0,
      params={
        "command_name": "twist",
        "base_height_target": 0.74,
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "smoothness": RewardTermCfg(
      func=homie_mdp.smoothness_second_order, weight=-0.05
    ),
    "joint_power": RewardTermCfg(
      func=homie_mdp.joint_power,
      weight=-2e-5,
      params={"command_name": "twist"},
    ),
    "feet_stumble": RewardTermCfg(
      func=homie_mdp.feet_stumble,
      weight=-1.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "torques": RewardTermCfg(func=homie_mdp.torques_l2, weight=-2.5e-6),
    "dof_vel": RewardTermCfg(func=homie_mdp.dof_vel_l2, weight=-1e-4),
    "dof_vel_limits": RewardTermCfg(
      func=homie_mdp.dof_vel_limits, weight=-2e-3, params={"soft_ratio": 0.80}
    ),
    "torque_limits": RewardTermCfg(
      func=homie_mdp.torque_limits, weight=-0.1, params={"soft_ratio": 0.95}
    ),
    "no_fly": RewardTermCfg(
      func=homie_mdp.no_fly,
      weight=0.75,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "joint_tracking_error": RewardTermCfg(
      func=homie_mdp.joint_tracking_error, weight=-0.05  # Homie scales
    ),
    "feet_slip": RewardTermCfg(
      func=homie_mdp.feet_slip_homie,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "feet_contact_forces": RewardTermCfg(
      func=homie_mdp.feet_contact_forces,
      weight=-0.00025,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 400.0},
    ),
    "contact_momentum": RewardTermCfg(
      func=homie_mdp.contact_momentum,
      weight=2.5e-4,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "action_vanish": RewardTermCfg(func=homie_mdp.action_vanish, weight=-1.0),
    "stand_still": RewardTermCfg(
      func=homie_mdp.stand_still,
      weight=-0.15,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "base_height_target": 0.74,
      },
    ),
    # Zero-velocity stand: base sway + waist joint rocking (vel/pose).
    "stand_sway": RewardTermCfg(
      func=homie_mdp.stand_sway,
      weight=-1.0,
      params={
        "command_name": "twist",
        "cmd_threshold": 0.1,
        "base_height_target": 0.74,
        "stand_height_threshold": 0.735,
        "waist_vel_weight": 1.0,
        "waist_pos_weight": 0.25,
        "waist_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
          ),
        ),
      },
    ),
  }

  # ---- Terminations ----
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  # ---- Curriculum (velocity ranges; upper-body ratio handled in events) ----
  cfg.curriculum = {
    "command_vel": CurriculumTermCfg(
      func=vel_mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.8, 1.2), "ang_vel_z": (-0.8, 0.8)},
          # 5000 iters * Homie num_steps_per_env=50
          {"step": 5000 * 50, "lin_vel_x": (-1.0, 1.5), "ang_vel_z": (-1.0, 1.0)},
        ],
      },
    ),
  }

  cfg.viewer.body_name = "torso_link"
  cfg.decimation = 4
  cfg.episode_length_s = 20.0

  # Remove height_scan obs leftovers if any remain in base cfg terms.
  for group in cfg.observations.values():
    group.terms.pop("height_scan", None)
    group.terms.pop("foot_height", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # Stop random resample / standing-zero fighting the Viser slider.
    # Keep ranges.height=(-0.5, 0) so the absolute-height slider stays movable.
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)
    twist.rel_standing_envs = 0.0

  return cfg
