#!/usr/bin/env python3
"""Phase-3 oracle play: phase-2 loco (99/14) + DLS IK, EE residual=0.

Sets ``HOMIE_PLAY_ORACLE_SUBGOAL=1`` so the Phase-3 *play* env keeps the
phase-2 observation/action layout (no ``99→229`` remap). No crate. DLS
chases the Viser grasp target (``p_sub := p_goal``); Arm chase default ON.

How to judge:
  - If the arm/waist reaches the gizmo with Arm chase ON → lower stack OK;
    Stage-3 actor / rate-limit / WM are the suspects.
  - If it still cannot reach → check ckpt, grasp ranges, Phase-3 IK interface.

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
      "Sets HOMIE_PLAY_ORACLE_SUBGOAL=1 → phase2 obs/act 99/14, no crate, "
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
    "[oracle-play] phase2 99/14 (no remap) + DLS; EE residual OFF; no crate\n"
    "  Expect: Actor in_features from 99-hist, NOT 'Remap 99->229'\n"
    "  1) Open Viser — robot should stand quietly first\n"
    "  2) Enable waist target + set XYZ gizmo / sliders\n"
    "  3) Arm chase default ON — watch if palm reaches the target\n"
    f"  argv: {argv}"
  )

  sys.argv = ["mjlab.scripts.play", *argv]
  from mjlab.scripts.play import main as play_main

  play_main()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
