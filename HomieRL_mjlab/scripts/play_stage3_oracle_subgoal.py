#!/usr/bin/env python3
"""Phase-3 oracle play (remap trial): phase-2 ckpt → 229/17 + DLS.

Sets ``HOMIE_PLAY_ORACLE_SUBGOAL=1``. Keeps Phase-3 obs/act layout so the
runner remaps ``99→229`` / ``14→17``. EE residual ignored; DLS chases Viser
grasp. Phase-3-only obs trails zeroed; Show container default OFF.

Example::

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
  # Must be set before importing homierl_mjlab (play_env_cfg built at register).
  os.environ["HOMIE_PLAY_ORACLE_SUBGOAL"] = "1"
  import homierl_mjlab  # noqa: F401

  argv = list(sys.argv[1:])
  if "--checkpoint-file" not in argv and "-h" not in argv and "--help" not in argv:
    print(
      "usage: play_stage3_oracle_subgoal.py --checkpoint-file <phase2.pt> "
      "[other mjlab.play flags]\n"
      "Sets HOMIE_PLAY_ORACLE_SUBGOAL=1 → Phase3 229/17 remap, no crate, "
      "EE Δp=0, Arm chase default ON.",
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
    "[oracle-play] remap 99->229 + slow DLS (crate OFF)\n"
    "  Expect: Remap phase-3 obs/act; Show container default OFF\n"
    "  1) Stand → Enable waist target → drag XYZ\n"
    "  2) Arm chase ON — palm tracks with play-grade IK\n"
    f"  argv: {argv}"
  )

  sys.argv = ["mjlab.scripts.play", *argv]
  from mjlab.scripts.play import main as play_main

  play_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
