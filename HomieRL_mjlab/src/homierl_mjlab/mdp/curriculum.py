"""Homie ``CurriculumScheduler`` (legged_gym.utils.curriculum)."""

from __future__ import annotations


class CurriculumScheduler:
  """Linear curriculum scheduler used by Homie arm/waist schedules."""

  def __init__(
    self,
    start_step: int,
    end_step: int,
    min_value: float = 0.0,
    max_value: float = 1.0,
    warmup: int = 1000,
    ramp: int = 8000,
  ):
    del end_step  # kept for Homie API compatibility
    self.start_step = int(start_step)
    self.min_value = float(min_value)
    self.max_value = float(max_value)
    self.warmup = int(warmup)
    self.ramp = int(ramp)

  def value(self, global_step: int) -> float:
    t = int(global_step) - self.start_step
    if t < 0:
      return 0.0
    if t < self.warmup:
      return self.min_value
    alpha = min(1.0, (t - self.warmup) / max(float(self.ramp), 1.0))
    return self.min_value + alpha * (self.max_value - self.min_value)
