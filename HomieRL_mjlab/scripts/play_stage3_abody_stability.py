#!/usr/bin/env python3
"""Play: Stage3 BPTT 17-D actor → verify a_body(14) loco.

Executes only the body head (``a_ee:=0``, arm chase OFF). Use to check whether
the trained dual-head policy's lower body is stable when actually applied
(train often keeps ``body_exec_teacher`` until arm unlock).

Example::

  .venv/bin/python scripts/play_stage3_abody_stability.py \\
    --bptt-checkpoint logs/rsl_rl/homie_g1/phase3/model_200.pt

  # Compare with frozen phase-2 teacher loco:
  .venv/bin/python scripts/play_stage3_abody_stability.py \\
    --loco-source teacher
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BPTT = _ROOT / "logs/rsl_rl/homie_g1/phase3/model_200.pt"


def main() -> int:
  argv = list(sys.argv[1:])

  if "--bptt-checkpoint" not in argv and "-h" not in argv and "--help" not in argv:
    if not _DEFAULT_BPTT.is_file():
      print(
        "usage: play_stage3_abody_stability.py "
        "--bptt-checkpoint <phase3/model_*.pt>\n"
        f"Missing default: {_DEFAULT_BPTT}",
        file=sys.stderr,
      )
      return 2
    argv = ["--bptt-checkpoint", str(_DEFAULT_BPTT), *argv]

  if "--ee-mode" not in argv:
    argv.extend(["--ee-mode", "actor"])
  if "--loco-source" not in argv:
    argv.extend(["--loco-source", "actor"])
  if "--body-only" not in argv:
    argv.append("--body-only")
  if "--curriculum" not in argv:
    argv.extend(["--curriculum", "A"])

  print(
    "[abody-play] Stage3 a_body(14) | a_ee:=0 | arm chase OFF\n"
    "  Flies ⇒ body head not ready (teacher-exec phase).\n"
    f"  → play_stage3_bptt.py {argv}"
  )
  sys.argv = [str(_ROOT / "scripts/play_stage3_bptt.py"), *argv]
  runpy.run_path(str(_ROOT / "scripts/play_stage3_bptt.py"), run_name="__main__")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
