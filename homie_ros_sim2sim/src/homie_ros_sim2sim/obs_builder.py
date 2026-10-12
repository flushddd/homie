"""Policy observation builder (Homie / MujocoDeploy layout)."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Sequence

import numpy as np

from homie_ros_sim2sim.math_utils import get_gravity_orientation


class HomieObsBuilder:
    """Build stacked observations matching Homie MujocoDeploy.

    Single-frame layout (n_body_joints = 29 typically)::

        cmd(3) | height(1) | omega(3) | gravity(3) |
        qj(n) | dqj(n) | last_action(num_actions_pad) | extras(16)

    ``num_actions_pad`` is usually 23 in Homie export (fixed slot).
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.n_joints = int(cfg["n_body_joints"])
        self.num_actions = int(cfg["num_actions"])
        self.action_pad = int(cfg.get("action_pad", 23))
        self.history_len = int(cfg.get("obs_history_len", 6))
        self.extras_dim = int(cfg.get("extras_dim", 16))

        self.single_obs_dim = (
            3 + 1 + 3 + 3 + self.n_joints + self.n_joints + self.action_pad + self.extras_dim
        )
        expected = cfg.get("num_obs")
        if expected is not None and int(expected) != self.single_obs_dim * self.history_len:
            raise ValueError(
                f"num_obs={expected} != single({self.single_obs_dim})*hist({self.history_len})"
            )

        self._history: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        z = np.zeros(self.single_obs_dim, dtype=np.float32)
        for _ in range(self.history_len):
            self._history.append(z.copy())

    def build_single(
        self,
        joint_q: np.ndarray,
        joint_dq: np.ndarray,
        quat_wxyz: np.ndarray,
        omega: np.ndarray,
        action: np.ndarray,
        cmd: Sequence[float],
        height_cmd: float,
        default_angles: np.ndarray,
        extras: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        qj = (joint_q - default_angles) * float(cfg["dof_pos_scale"])
        dqj = joint_dq * float(cfg["dof_vel_scale"])
        grav = get_gravity_orientation(quat_wxyz).astype(np.float32)
        omega_s = omega * float(cfg["ang_vel_scale"])

        act = np.zeros(self.action_pad, dtype=np.float32)
        n = min(len(action), self.action_pad)
        act[:n] = np.asarray(action[:n], dtype=np.float32)

        ex = np.zeros(self.extras_dim, dtype=np.float32)
        if extras is not None:
            m = min(len(extras), self.extras_dim)
            ex[:m] = np.asarray(extras[:m], dtype=np.float32)
        else:
            # Flags compatible with MujocoDeploy g1.yaml
            ex[0] = float(cfg.get("is_target", 0))
            ex[1] = float(cfg.get("is_waist", 0))
            ex[2] = float(cfg.get("extend_dist", 0.0))
            ex[3] = float(cfg.get("is_target_height", 0))

        obs = np.zeros(self.single_obs_dim, dtype=np.float32)
        cmd_s = np.asarray(cmd[:3], dtype=np.float32) * np.asarray(
            cfg["cmd_scale"], dtype=np.float32
        )
        obs[0:3] = cmd_s
        obs[3:4] = height_cmd
        obs[4:7] = omega_s
        obs[7:10] = grav
        i0 = 10
        obs[i0 : i0 + self.n_joints] = qj
        i0 += self.n_joints
        obs[i0 : i0 + self.n_joints] = dqj
        i0 += self.n_joints
        obs[i0 : i0 + self.action_pad] = act
        i0 += self.action_pad
        obs[i0 : i0 + self.extras_dim] = ex
        return obs

    def push(self, single: np.ndarray) -> np.ndarray:
        self._history.append(np.asarray(single, dtype=np.float32).copy())
        return np.concatenate(list(self._history), axis=0)
