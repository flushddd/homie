"""Small math helpers shared by sim and policy nodes."""

from __future__ import annotations

import numpy as np


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by inverse of unit quaternion ``q`` (wxyz)."""
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame from base quaternion (wxyz)."""
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))


def pd_torque(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Classic PD torque: kp*(q_des-q) + kd*(dq_des-dq)."""
    return kp * (target_q - q) + kd * (target_dq - dq)
