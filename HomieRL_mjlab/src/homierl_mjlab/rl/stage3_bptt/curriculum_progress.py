"""Progressive Stage-3 curriculum: body → arm → obstacles.

Phases
------
0 BODY: arm locked (δ=0, p_sub=palm); presence=0; teacher BC on loco.
1 ARM:  arm unlocked; presence=0; unlock when body stable.
2 BOX:  arm unlocked; presence ramps when clear-reach is good enough.

Legacy ``curriculum=A|B`` still works when ``curriculum_auto=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PHASE_BODY = 0
PHASE_ARM = 1
PHASE_BOX = 2
PHASE_NAMES = ("body", "arm", "box")


@dataclass
class ProgressiveCurriculum:
  """EMA-gated phase machine for ``body_ee_palm_delta`` training."""

  # --- gates (engineering defaults) ---
  ema_beta: float = 0.9
  # BODY → ARM
  body_min_iters: int = 30
  body_hold_iters: int = 5
  teacher_mse_max: float = 0.08
  ep_len_frac_min: float = 0.8  # vs max episode length (policy steps)
  # ARM → BOX
  arm_min_iters: int = 40
  arm_hold_iters: int = 5
  reach_unlock_box: float = 0.50
  # BOX presence ramp
  box_min_iters_between_bumps: int = 20
  box_hold_iters: int = 5
  clear_reach_bump: float = 0.30
  presence_step: float = 0.05
  presence_max: float = 0.35

  # --- live state ---
  phase: int = PHASE_BODY
  arm_unlocked: bool = False
  presence_prob: float = 0.0
  phase_iters: int = 0
  hold_ok: int = 0
  iters_since_bump: int = 0
  max_ep_len: float = 500.0

  reach_ema: float = 0.0
  clear_reach_ema: float = 0.0
  teacher_mse_ema: float = 1.0
  ep_len_ema: float = 0.0
  _ema_inited: bool = False
  events: list[str] = field(default_factory=list)

  @classmethod
  def from_cfg(cls, cfg) -> ProgressiveCurriculum:
    return cls(
      ema_beta=float(getattr(cfg, "curr_ema_beta", 0.9)),
      body_min_iters=int(getattr(cfg, "curr_body_min_iters", 30)),
      body_hold_iters=int(getattr(cfg, "curr_body_hold_iters", 5)),
      teacher_mse_max=float(getattr(cfg, "curr_teacher_mse_max", 0.08)),
      ep_len_frac_min=float(getattr(cfg, "curr_ep_len_frac_min", 0.8)),
      arm_min_iters=int(getattr(cfg, "curr_arm_min_iters", 40)),
      arm_hold_iters=int(getattr(cfg, "curr_arm_hold_iters", 5)),
      reach_unlock_box=float(getattr(cfg, "curr_reach_unlock_box", 0.50)),
      box_min_iters_between_bumps=int(
        getattr(cfg, "curr_box_min_iters_between_bumps", 20)
      ),
      box_hold_iters=int(getattr(cfg, "curr_box_hold_iters", 5)),
      clear_reach_bump=float(getattr(cfg, "curr_clear_reach_bump", 0.30)),
      presence_step=float(getattr(cfg, "curr_presence_step", 0.05)),
      presence_max=float(
        getattr(cfg, "curr_presence_max", getattr(cfg, "curriculum_b_presence_prob", 0.35))
      ),
      phase=PHASE_BODY,
      arm_unlocked=False,
      presence_prob=0.0,
    )

  @property
  def phase_name(self) -> str:
    return PHASE_NAMES[int(self.phase)]

  def _ema(self, prev: float, x: float) -> float:
    b = float(self.ema_beta)
    if not self._ema_inited:
      return float(x)
    return b * float(prev) + (1.0 - b) * float(x)

  def observe(self, logs: dict[str, float]) -> None:
    reach = float(logs.get("reach_success_frac", 0.0))
    clear = float(logs.get("clear_reach_success_frac", 0.0))
    tmse = float(logs.get("teacher_mse_collect", logs.get("teacher_mse", 1.0)))
    ep_len = float(logs.get("mean_episode_length", 0.0))
    if tmse != tmse:
      tmse = self.teacher_mse_ema
    self.reach_ema = self._ema(self.reach_ema, reach)
    self.clear_reach_ema = self._ema(self.clear_reach_ema, clear)
    self.teacher_mse_ema = self._ema(self.teacher_mse_ema, tmse)
    if ep_len > 0.0:
      self.ep_len_ema = self._ema(self.ep_len_ema, ep_len)
    self._ema_inited = True

  def step(self, logs: dict[str, float]) -> dict[str, float]:
    """Update EMAs and maybe advance phase / bump presence. Returns status dict."""
    self.events.clear()
    self.observe(logs)
    self.phase_iters += 1
    self.iters_since_bump += 1

    if self.phase == PHASE_BODY:
      self._try_unlock_arm()
    elif self.phase == PHASE_ARM:
      self._try_enter_box()
    elif self.phase == PHASE_BOX:
      self._try_bump_presence()

    return self.status_dict()

  def _try_unlock_arm(self) -> None:
    ep_ok = True
    if self.max_ep_len > 1.0 and self.ep_len_ema > 0.0:
      ep_ok = self.ep_len_ema >= self.ep_len_frac_min * self.max_ep_len
    mse_ok = self.teacher_mse_ema <= self.teacher_mse_max
    ready = (
      self.phase_iters >= self.body_min_iters
      and ep_ok
      and mse_ok
    )
    if ready:
      self.hold_ok += 1
    else:
      self.hold_ok = 0
    if self.hold_ok >= self.body_hold_iters:
      self.phase = PHASE_ARM
      self.arm_unlocked = True
      self.presence_prob = 0.0
      self.phase_iters = 0
      self.hold_ok = 0
      self.iters_since_bump = 0
      self.events.append(
        f"unlock_arm teacher_mse_ema={self.teacher_mse_ema:.4f} "
        f"ep_len_ema={self.ep_len_ema:.1f}"
      )

  def _try_enter_box(self) -> None:
    ready = (
      self.phase_iters >= self.arm_min_iters
      and self.reach_ema >= self.reach_unlock_box
    )
    if ready:
      self.hold_ok += 1
    else:
      self.hold_ok = 0
    if self.hold_ok >= self.arm_hold_iters:
      self.phase = PHASE_BOX
      self.arm_unlocked = True
      # First bump immediately to a small presence so crates start appearing.
      self.presence_prob = min(self.presence_step, self.presence_max)
      self.phase_iters = 0
      self.hold_ok = 0
      self.iters_since_bump = 0
      self.events.append(
        f"enter_box reach_ema={self.reach_ema:.3f} "
        f"presence={self.presence_prob:.2f}"
      )

  def _try_bump_presence(self) -> None:
    if self.presence_prob >= self.presence_max - 1e-9:
      return
    ready = (
      self.iters_since_bump >= self.box_min_iters_between_bumps
      and self.clear_reach_ema >= self.clear_reach_bump
    )
    if ready:
      self.hold_ok += 1
    else:
      self.hold_ok = 0
    if self.hold_ok >= self.box_hold_iters:
      old = self.presence_prob
      self.presence_prob = min(
        self.presence_prob + self.presence_step, self.presence_max
      )
      self.hold_ok = 0
      self.iters_since_bump = 0
      self.events.append(
        f"presence {old:.2f}→{self.presence_prob:.2f} "
        f"clear_reach_ema={self.clear_reach_ema:.3f}"
      )

  def status_dict(self) -> dict[str, float]:
    return {
      "curr_phase": float(self.phase),
      "curr_arm_unlocked": 1.0 if self.arm_unlocked else 0.0,
      "curr_presence_prob": float(self.presence_prob),
      "curr_phase_iters": float(self.phase_iters),
      "curr_hold_ok": float(self.hold_ok),
      "curr_reach_ema": float(self.reach_ema),
      "curr_clear_reach_ema": float(self.clear_reach_ema),
      "curr_teacher_mse_ema": float(self.teacher_mse_ema),
      "curr_ep_len_ema": float(self.ep_len_ema),
    }

  def state_dict(self) -> dict:
    return {
      "phase": int(self.phase),
      "arm_unlocked": bool(self.arm_unlocked),
      "presence_prob": float(self.presence_prob),
      "phase_iters": int(self.phase_iters),
      "hold_ok": int(self.hold_ok),
      "iters_since_bump": int(self.iters_since_bump),
      "max_ep_len": float(self.max_ep_len),
      "reach_ema": float(self.reach_ema),
      "clear_reach_ema": float(self.clear_reach_ema),
      "teacher_mse_ema": float(self.teacher_mse_ema),
      "ep_len_ema": float(self.ep_len_ema),
      "_ema_inited": bool(self._ema_inited),
    }

  def load_state_dict(self, sd: dict) -> None:
    self.phase = int(sd.get("phase", PHASE_BODY))
    self.arm_unlocked = bool(sd.get("arm_unlocked", self.phase >= PHASE_ARM))
    self.presence_prob = float(sd.get("presence_prob", 0.0))
    self.phase_iters = int(sd.get("phase_iters", 0))
    self.hold_ok = int(sd.get("hold_ok", 0))
    self.iters_since_bump = int(sd.get("iters_since_bump", 0))
    self.max_ep_len = float(sd.get("max_ep_len", self.max_ep_len))
    self.reach_ema = float(sd.get("reach_ema", 0.0))
    self.clear_reach_ema = float(sd.get("clear_reach_ema", 0.0))
    self.teacher_mse_ema = float(sd.get("teacher_mse_ema", 1.0))
    self.ep_len_ema = float(sd.get("ep_len_ema", 0.0))
    self._ema_inited = bool(sd.get("_ema_inited", False))
