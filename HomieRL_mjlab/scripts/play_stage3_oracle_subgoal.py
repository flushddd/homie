#!/usr/bin/env python3
"""Phase-3 oracle play: p_sub := p_goal via standard mjlab play + phase-2 loco.

No crate. EE residual forced to 0. Frozen loco+DLS tracks the Viser XYZ goal.
Use a phase-2 (or phase-3) checkpoint — same lower stack as BPTT.

How to judge:
  - If the arm/waist reaches the gizmo with Arm chase ON → lower stack OK;
    Stage-3 actor / rate-limit / WM are the suspects.
  - If it still cannot reach → check ckpt, obs remap, grasp ranges, Phase-3 interface.

Example::

  HOMIE_PLAY_ORACLE_SUBGOAL=1 .venv/bin/python -m mjlab.scripts.play \\
    Mjlab-Homie-Flat-Unitree-G1-Phase3 --num-envs 1 --viewer viser \\
    --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18600.pt

Or this wrapper (sets the env var for you)::

  .venv/bin/python scripts/play_stage3_oracle_subgoal.py \\
    --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18600.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
  os.environ["HOMIE_PLAY_ORACLE_SUBGOAL"] = "1"
  # Ensure Homie tasks are registered before mjlab.play parses the task id.
  import homierl_mjlab  # noqa: F401

  argv = list(sys.argv[1:])
  if "--checkpoint-file" not in argv and "-h" not in argv and "--help" not in argv:
    print(
      "usage: play_stage3_oracle_subgoal.py --checkpoint-file <phase2.pt> "
      "[other mjlab.play flags]\n"
      "Sets HOMIE_PLAY_ORACLE_SUBGOAL=1 → no crate, EE Δp=0, Arm chase default ON.",
      file=sys.stderr,
    )
    return 2

  if not any(a.startswith("Mjlab-") for a in argv):
    argv = [
      "Mjlab-Homie-Flat-Unitree-G1-Phase3",
      "--num-envs",
      "1",
      "--viewer",
      "viser",
      *argv,
    ]

  print(
    "[oracle-play] p_sub := p_goal (EE residual OFF, no container)\n"
    "  1) Open Viser URL printed below\n"
    "  2) Enable waist target + set XYZ gizmo / sliders\n"
    "  3) Arm chase should already be ON — watch if palm reaches the target\n"
    f"  argv: {argv}"
  )

  sys.argv = ["mjlab.scripts.play", *argv]
  from mjlab.scripts.play import main as play_main

  play_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
