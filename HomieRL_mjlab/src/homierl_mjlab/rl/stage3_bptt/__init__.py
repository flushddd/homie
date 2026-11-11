"""Stage-3 BPTT + world-model training (parallel to PPO Phase-3).

See ``INTERFACES.md`` for dims and limitations. Existing task
``Mjlab-Homie-Flat-Unitree-G1-Phase3`` (PPO) is unchanged.
"""

from homierl_mjlab.rl.stage3_bptt.cfg import Stage3BpttCfg
from homierl_mjlab.rl.stage3_bptt.checks import run_all_checks
from homierl_mjlab.rl.stage3_bptt.collector import Stage3SimCollector
from homierl_mjlab.rl.stage3_bptt.dls import dls_step, dls_toward
from homierl_mjlab.rl.stage3_bptt.networks import Stage3Actor, Stage3Critic, WorldModel
from homierl_mjlab.rl.stage3_bptt.replay import EpisodeReplay
from homierl_mjlab.rl.stage3_bptt.trainer import Stage3BpttTrainer

__all__ = [
  "EpisodeReplay",
  "Stage3Actor",
  "Stage3BpttCfg",
  "Stage3BpttTrainer",
  "Stage3Critic",
  "Stage3SimCollector",
  "WorldModel",
  "dls_step",
  "dls_toward",
  "run_all_checks",
]
