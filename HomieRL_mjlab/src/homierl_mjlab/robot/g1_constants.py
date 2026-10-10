"""Unitree G1 (Inspire-hand waist) constants for HOMIE mjlab training.

Loads HomieRL ``g1_inspire_description_waist`` MJCF, then patches it so mjlab
velocity/HOMIE MDP terms work.

Phase-1 control:
- All 29 hinges stay in the model (obs dim ready for later phases).
- Policy actions = 12 legs only.
- Waist + arms have PD (Homie kp/kd) but hold the **default / init** joint
  angles (no policy action, no curriculum).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  ACTUATOR_4010,
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
  G1_ACTION_SCALE,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Absolute path to HomieRL inspire-waist description.
_INSPIRE_WAIST_DIR = Path(
  "/home/flush/OpenHomie/HomieRL/legged_gym/resources/robots/"
  "g1_inspire_description_waist"
)
INSPIRE_WAIST_XML = (
  _INSPIRE_WAIST_DIR / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
)

# Held at default pose by PD (no policy action in phase-1).
WAIST_JOINT_NAMES = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)

ARM_JOINT_NAMES = (
  ".*_shoulder_pitch_joint",
  ".*_shoulder_roll_joint",
  ".*_shoulder_yaw_joint",
  ".*_elbow_joint",
  ".*_wrist_roll_joint",
  ".*_wrist_pitch_joint",
  ".*_wrist_yaw_joint",
)

# Homie ``g1_inspire_wiast_config.init_state.default_joint_angles`` (1:1).
# Also the PD hold / action=0 target via ``default_joint_pos``.
HOMIE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.75),
  joint_pos={
    # legs
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # left arm
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # right arm
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # inspire fingers (present if XML exposes them; ignored otherwise)
    "left_thumb_1_joint": 0.0,
    "left_thumb_2_joint": 0.0,
    "left_thumb_3_joint": 0.0,
    "left_thumb_4_joint": 0.0,
    "left_index_1_joint": 0.0,
    "left_index_2_joint": 0.0,
    "left_middle_1_joint": 0.0,
    "left_middle_2_joint": 0.0,
    "left_ring_1_joint": 0.0,
    "left_ring_2_joint": 0.0,
    "left_little_1_joint": 0.0,
    "left_little_2_joint": 0.0,
    "right_thumb_1_joint": 0.0,
    "right_thumb_2_joint": 0.0,
    "right_thumb_3_joint": 0.0,
    "right_thumb_4_joint": 0.0,
    "right_index_1_joint": 0.0,
    "right_index_2_joint": 0.0,
    "right_middle_1_joint": 0.0,
    "right_middle_2_joint": 0.0,
    "right_ring_1_joint": 0.0,
    "right_ring_2_joint": 0.0,
    "right_little_1_joint": 0.0,
    "right_little_2_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# Policy actions: 12 lower-limb joints only.
LEG_JOINT_NAMES = (
  ".*_hip_yaw_joint",
  ".*_hip_roll_joint",
  ".*_hip_pitch_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
)

UPPER_JOINT_NAMES = WAIST_JOINT_NAMES + ARM_JOINT_NAMES

KNEE_JOINT_NAMES = (".*_knee_joint",)

# Homie action_scale = 0.25 on default joint offsets.
HOMIE_LEG_ACTION_SCALE: float = 0.25

# Homie ``g1_inspire_wiast_config.control`` stiffness / damping [N·m/rad].
# Grouped by identical (kp, kd); effort/armature keep Unitree motor specs.
HOMIE_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
HOMIE_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
HOMIE_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
HOMIE_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
HOMIE_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=40.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
HOMIE_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
  ),
  stiffness=300.0,
  damping=5.0,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
HOMIE_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=200.0,
  damping=2.0,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
HOMIE_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint",),
  stiffness=100.0,
  damping=1.0,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
HOMIE_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_roll_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
HOMIE_ACTUATOR_WRIST_PY = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Legs (policy) + waist/arms (PD hold at default). Hands stay unactuated.
HOMIE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HOMIE_ACTUATOR_HIP_YAW,
    HOMIE_ACTUATOR_HIP_PITCH,
    HOMIE_ACTUATOR_HIP_ROLL,
    HOMIE_ACTUATOR_KNEE,
    HOMIE_ACTUATOR_ANKLE,
    HOMIE_ACTUATOR_WAIST,
    HOMIE_ACTUATOR_SHOULDER,
    HOMIE_ACTUATOR_ELBOW,
    HOMIE_ACTUATOR_WRIST_ROLL,
    HOMIE_ACTUATOR_WRIST_PY,
  ),
  # Homie ``rewards.soft_dof_pos_limit = 0.975``
  soft_joint_pos_limit_factor=0.975,
)

# Foot sphere collisions from the inspire XML (4 per foot after naming).
FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-4]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  disable_other_geoms=True,
)


def _ensure_named_sensor(
  spec: mujoco.MjSpec,
  name: str,
  sensor_type: mujoco.mjtSensor,
  site_name: str,
) -> None:
  existing = {s.name for s in spec.sensors}
  if name in existing:
    return
  spec.add_sensor(
    name=name,
    type=sensor_type,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname=site_name,
  )


def _patch_inspire_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Make Homie inspire MJCF compatible with mjlab HOMIE env terms."""
  # Resolve meshes relative to the XML directory regardless of CWD.
  spec.meshdir = str(_INSPIRE_WAIST_DIR / "meshes")

  # Drop XML <motor> actuators; HOMIE_ARTICULATION installs BuiltinPosition PD.
  for act in list(spec.actuators):
    spec.delete(act)

  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    spheres = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
    for i, geom in enumerate(spheres, start=1):
      geom.name = f"{side}_foot{i}_collision"
    site_names = {s.name for s in body.sites}
    if f"{side}_foot" not in site_names:
      body.add_site(
        name=f"{side}_foot",
        pos=[0.02, 0.0, -0.03],
        size=[0.01, 0.0, 0.0],
        rgba=[1.0, 0.0, 0.0, 1.0],
      )

  _ensure_named_sensor(
    spec, "imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO, "imu_in_pelvis"
  )
  _ensure_named_sensor(
    spec, "imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER, "imu_in_pelvis"
  )
  return spec


def get_inspire_waist_spec() -> mujoco.MjSpec:
  if not INSPIRE_WAIST_XML.is_file():
    raise FileNotFoundError(f"Inspire waist MJCF not found: {INSPIRE_WAIST_XML}")
  spec = mujoco.MjSpec.from_file(str(INSPIRE_WAIST_XML))
  return _patch_inspire_spec(spec)


def get_g1_robot_cfg() -> EntityCfg:
  """G1 inspire-waist: 29-DoF obs; PD on legs+waist+arms; policy = 12 legs."""
  return EntityCfg(
    init_state=HOMIE_INIT_STATE,
    collisions=(FEET_COLLISION,),
    spec_fn=get_inspire_waist_spec,
    articulation=HOMIE_ARTICULATION,
  )


FOOT_COLLISION_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
)

__all__ = [
  "ARM_JOINT_NAMES",
  "FEET_COLLISION",
  "FOOT_COLLISION_GEOM_NAMES",
  "G1_ACTION_SCALE",
  "HOMIE_ARTICULATION",
  "HOMIE_INIT_STATE",
  "HOMIE_LEG_ACTION_SCALE",
  "INSPIRE_WAIST_XML",
  "KNEE_JOINT_NAMES",
  "LEG_JOINT_NAMES",
  "UPPER_JOINT_NAMES",
  "WAIST_JOINT_NAMES",
  "get_g1_robot_cfg",
  "get_inspire_waist_spec",
]
