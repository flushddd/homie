"""Stage-1/2 SuperMimic BPTT experiment (parallel to HIM-PPO and Stage-3).

See ``README.md``. Does not modify ``stage3_bptt`` or phase-1/2 PPO tasks.
"""

from homierl_mjlab.rl.stage12_bptt.cfg import Stage12BpttCfg
from homierl_mjlab.rl.stage12_bptt.checks import run_all_checks
from homierl_mjlab.rl.stage12_bptt.collector import Stage12SimCollector
from homierl_mjlab.rl.stage12_bptt.networks import LocoActor, LocoCritic, WorldModel
from homierl_mjlab.rl.stage12_bptt.trainer import Stage12BpttTrainer

__all__ = [
  "LocoActor",
  "LocoCritic",
  "Stage12BpttCfg",
  "Stage12BpttTrainer",
  "Stage12SimCollector",
  "WorldModel",
  "run_all_checks",
]
