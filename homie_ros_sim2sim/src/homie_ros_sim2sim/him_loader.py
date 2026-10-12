"""Load HomieRL_mjlab HIMActorCritic from an rsl_rl ``model_*.pt`` checkpoint."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch


def _ensure_mjlab_on_path() -> None:
    candidates = [
        os.environ.get("HOMIERL_MJLAB_SRC", ""),
        os.path.expanduser("~/OpenHomie/HomieRL_mjlab/src"),
        "/home/flush/OpenHomie/HomieRL_mjlab/src",
    ]
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


def load_him_policy(
    checkpoint_path: str,
    *,
    num_actions: int,
    num_one_step_obs: int,
    actor_history_length: int = 6,
    device: str = "cpu",
) -> Tuple[Any, int]:
    """Return (actor_critic in eval mode, one_step_dim)."""
    _ensure_mjlab_on_path()
    from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic

    num_actor_obs = num_one_step_obs * actor_history_length
    # Infer critic input width from checkpoint (unused at act_inference).
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    critic_in = int(sd["critic.0.weight"].shape[1]) if "critic.0.weight" in sd else (
        num_one_step_obs + 3
    )

    ac = HIMActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=critic_in,
        num_one_step_obs=num_one_step_obs,
        num_one_step_critic_obs=critic_in,
        actor_history_length=actor_history_length,
        critic_history_length=1,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    missing, unexpected = ac.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if "optimizer" not in k]
    if unexpected:
        print(f"[him_loader] unexpected keys: {unexpected[:8]}")
    if missing:
        print(f"[him_loader] missing keys: {missing[:8]}")
    ac.to(device)
    ac.eval()
    return ac, num_one_step_obs


class HimInference:
    """Thin wrapper: one-step numpy obs → action mean."""

    def __init__(self, actor_critic: Any, one_step: int, hist_len: int, device: str = "cpu"):
        self.ac = actor_critic
        self.one_step = one_step
        self.hist_len = hist_len
        self.device = device
        self._hist: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._hist = None

    def __call__(self, step: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(step, dtype=np.float32)).view(1, -1).to(self.device)
        if self._hist is None:
            self._hist = t.repeat(1, self.hist_len)
        else:
            self._hist = torch.cat([self._hist[:, self.one_step :], t], dim=-1)
        with torch.inference_mode():
            act = self.ac.act_inference(self._hist)
        return act.squeeze(0).cpu().numpy().astype(np.float32)
