"""Build one-step actor obs matching HomieRL_mjlab ``env_cfgs`` phase-1/2."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from homie_ros_sim2sim.math_utils import get_gravity_orientation


def build_mjlab_one_step(
    joint_q: np.ndarray,
    joint_dq: np.ndarray,
    quat_wxyz: np.ndarray,
    omega: np.ndarray,
    action: np.ndarray,
    cmd_xy_yaw: Sequence[float],
    height_cmd: float,
    default_angles: np.ndarray,
    *,
    cmd_scale: Sequence[float] = (2.0, 2.0, 0.5),
    ang_vel_scale: float = 0.5,
    dof_pos_scale: float = 1.0,
    dof_vel_scale: float = 0.05,
    grasp_extras: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Phase-1 one-step = 80; phase-2 appends grasp trail after actions."""
    qj = (joint_q - default_angles) * dof_pos_scale
    dqj = joint_dq * dof_vel_scale
    grav = get_gravity_orientation(quat_wxyz)
    omega_s = omega * ang_vel_scale
    cmd = np.asarray(cmd_xy_yaw[:3], dtype=np.float64) * np.asarray(cmd_scale, dtype=np.float64)

    parts = [
        cmd.astype(np.float32),
        np.array([height_cmd], dtype=np.float32),
        omega_s.astype(np.float32),
        grav.astype(np.float32),
        qj.astype(np.float32),
        dqj.astype(np.float32),
        np.asarray(action, dtype=np.float32).reshape(-1),
    ]
    if grasp_extras is not None and len(grasp_extras) > 0:
        parts.append(np.asarray(grasp_extras, dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


class HistoryBuffer:
    """Rolling proprioceptive history for HIM (hist_len × one_step)."""

    def __init__(self, hist_len: int, one_step: int):
        self.hist_len = hist_len
        self.one_step = one_step
        self._hist: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._hist = None

    def push(self, step: np.ndarray) -> np.ndarray:
        step = np.asarray(step, dtype=np.float32).reshape(-1)
        if step.shape[0] != self.one_step:
            raise ValueError(f"one-step dim {step.shape[0]} != {self.one_step}")
        if self._hist is None:
            self._hist = np.tile(step, self.hist_len)
        else:
            self._hist = np.concatenate([self._hist[self.one_step :], step], axis=0)
        return self._hist
