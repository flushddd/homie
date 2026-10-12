"""Resolve MuJoCo joint / actuator indices by name."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np


# Homie / mjlab body DOF order used for sensors & PD (29 hinges).
G1_BODY_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Phase-1 policy actions (12 legs). Order matches Homie default_angles legs.
PHASE1_ACTION_JOINTS: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

# Phase-2: legs + waist roll/pitch (yaw held at default).
PHASE2_ACTION_JOINTS: Tuple[str, ...] = PHASE1_ACTION_JOINTS + (
    "waist_roll_joint",
    "waist_pitch_joint",
)


def resolve_joint_qadr(model: mujoco.MjModel, names: Sequence[str]) -> List[int]:
    """Return qpos addresses for named hinge joints."""
    addrs: List[int] = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found in MJCF: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return addrs


def resolve_joint_dadr(model: mujoco.MjModel, names: Sequence[str]) -> List[int]:
    """Return qvel/dof addresses for named hinge joints."""
    addrs: List[int] = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"Joint not found in MJCF: {name}")
        addrs.append(int(model.jnt_dofadr[jid]))
    return addrs


def resolve_actuator_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> List[int]:
    """Map joint names → actuator ids (actuator that drives that joint)."""
    ids: List[int] = []
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise KeyError(f"Joint not found: {jname}")
        found = -1
        for aid in range(model.nu):
            # trnid[aid, 0] is joint id for joint actuators
            if int(model.actuator_trnid[aid, 0]) == jid:
                found = aid
                break
        if found < 0:
            # fallback: actuator named like the joint
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if aid < 0:
                raise KeyError(f"No actuator for joint: {jname}")
            found = aid
        ids.append(int(found))
    return ids


def build_default_q_map(default_angles: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in default_angles.items()}


def pack_joint_vector(
    names: Sequence[str],
    values: Dict[str, float],
    fill: float = 0.0,
) -> np.ndarray:
    return np.array([values.get(n, fill) for n in names], dtype=np.float64)
