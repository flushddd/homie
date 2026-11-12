#!/usr/bin/env python3
"""Play: verify Stage3 BPTT a_body(14) — EE Δp forced to 0.

Always sets ``HOMIE_PLAY_ORACLE_SUBGOAL=1`` (standing, crate OFF, slow DLS)
so the env matches oracle play UX. Stage3 actor only drives ``a_body``;
``a_ee`` is zeroed. Two modes:

  * ``body`` (default): ``p_sub:=palm``, arm chase OFF — loco / stand stability.
  * ``body+ik``: ``p_sub:=p_goal``, Arm chase after waist target — whole-body
    quality of Stage3 a_body + DLS vs oracle (phase-2 loco + DLS).

Examples::

  .venv/bin/python scripts/play_stage3_abody_stability.py \\
    --bptt-checkpoint logs/rsl_rl/homie_g1/phase3/model_200.pt

  # With arm IK (compare to play_stage3_oracle_subgoal.py):
  .venv/bin/python scripts/play_stage3_abody_stability.py \\
    --bptt-checkpoint logs/.../model_200.pt --mode body+ik

  # Teacher loco baseline under the same EE=0 setup:
  .venv/bin/python scripts/play_stage3_abody_stability.py \\
    --loco-source teacher --mode body+ik
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BPTT = _ROOT / "logs/rsl_rl/homie_g1/phase3/model_200.pt"


def _parse_wrapper(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
  p = argparse.ArgumentParser(
    description="Stage3 a_body stability play (wraps play_stage3_bptt.py)",
    allow_abbrev=False,
  )
  p.add_argument(
    "--mode",
    type=str,
    default="body",
    choices=("body", "body+ik"),
    help=(
      "body: a_ee=0, p_sub=palm, arm OFF (stability). "
      "body+ik: a_ee=0, p_sub=p_goal, Arm chase gated (vs oracle)."
    ),
  )
  p.add_argument(
    "--bptt-checkpoint",
    type=str,
    default="",
    help="Stage3 BPTT ckpt (default: logs/.../phase3/model_200.pt if present)",
  )
  known, rest = p.parse_known_args(argv)
  return known, rest


def main() -> int:
  # Must be set before play_stage3_bptt imports / builds Phase3 play_env_cfg.
  os.environ["HOMIE_PLAY_ORACLE_SUBGOAL"] = "1"

  known, rest = _parse_wrapper(sys.argv[1:])
  argv = list(rest)

  bptt = str(known.bptt_checkpoint or "").strip()
  if not bptt and "--bptt-checkpoint" not in argv:
    if "-h" not in argv and "--help" not in argv:
      if not _DEFAULT_BPTT.is_file():
        print(
          "usage: play_stage3_abody_stability.py "
          "--bptt-checkpoint <phase3/model_*.pt> [--mode body|body+ik]\n"
          f"Missing default: {_DEFAULT_BPTT}",
          file=sys.stderr,
        )
        return 2
      bptt = str(_DEFAULT_BPTT)
  if bptt:
    argv = ["--bptt-checkpoint", bptt, *argv]

  if "--ee-mode" not in argv:
    argv.extend(["--ee-mode", "actor"])
  if "--loco-source" not in argv:
    argv.extend(["--loco-source", "actor"])
  if "--curriculum" not in argv:
    argv.extend(["--curriculum", "A"])

  mode = str(known.mode)
  # Strip conflicting flags if user passed them; wrapper owns EE pinning.
  strip = {"--body-only", "--zero-ee-delta"}
  cleaned: list[str] = []
  skip_next = False
  for i, a in enumerate(argv):
    if skip_next:
      skip_next = False
      continue
    if a in strip:
      continue
    if a == "--subgoal":
      skip_next = True
      continue
    if a.startswith("--subgoal="):
      continue
    cleaned.append(a)
  argv = cleaned

  if mode == "body":
    argv.append("--body-only")
    banner = (
      "[abody-play] mode=body | Stage3 a_body(14) | a_ee:=0 | p_sub:=palm | "
      "arm chase OFF\n"
      "  Oracle env (standing / crate OFF). Flies ⇒ body head not ready.\n"
      "  Compare loco: --loco-source teacher"
    )
  else:
    argv.extend(["--zero-ee-delta", "--subgoal", "goal"])
    banner = (
      "[abody-play] mode=body+ik | Stage3 a_body(14) | a_ee:=0 | p_sub:=p_goal\n"
      "  Same env as oracle: Enable waist target → XYZ → Arm chase ON.\n"
      "  Expect quality near play_stage3_oracle_subgoal (phase-2 loco + DLS).\n"
      "  Teacher loco under same IK: --loco-source teacher"
    )

  print(f"{banner}\n  → play_stage3_bptt.py {argv}")
  sys.argv = [str(_ROOT / "scripts/play_stage3_bptt.py"), *argv]
  runpy.run_path(str(_ROOT / "scripts/play_stage3_bptt.py"), run_name="__main__")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
