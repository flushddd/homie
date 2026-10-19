#!/usr/bin/env python3
"""Archive an existing HomieRL_mjlab training run (no train / no sim).

Usage examples::

  # Absolute or relative run directory
  python scripts/experiment_archive/archive_train_run.py \\
    logs/rsl_rl/homie_g1/2026-10-17_15-05-22_phase2

  # Shorthand run folder name under default logs root
  python scripts/experiment_archive/archive_train_run.py 2026-10-17_15-05-22_phase2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from archive_lib import archive_train_run, resolve_run_dir  # noqa: E402


def _default_roots() -> tuple[Path, Path, Path]:
  # .../HomieRL_mjlab/scripts/experiment_archive/this_file.py
  mjlab_root = Path(__file__).resolve().parents[2]
  repo_root = mjlab_root.parent  # OpenHomie
  logs_root = mjlab_root / "logs" / "rsl_rl" / "homie_g1"
  archives_root = mjlab_root / "experiments" / "archives"
  return repo_root, logs_root, archives_root


def main(argv: list[str] | None = None) -> int:
  repo_root, logs_root, archives_root = _default_roots()
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "run",
    type=Path,
    help="Training run directory (must contain params/env.yaml + agent.yaml)",
  )
  p.add_argument(
    "--archives-root",
    type=Path,
    default=archives_root,
    help=f"Archive output root (default: {archives_root})",
  )
  p.add_argument(
    "--repo-root",
    type=Path,
    default=repo_root,
    help=f"Git repo root for version snapshot (default: {repo_root})",
  )
  p.add_argument(
    "--logs-root",
    type=Path,
    default=logs_root,
    help=f"Default logs root for shorthand names (default: {logs_root})",
  )
  p.add_argument(
    "--archive-id",
    type=str,
    default=None,
    help="Optional archive folder name; default is <timestamp>__<run_name>",
  )
  args = p.parse_args(argv)

  run_dir = resolve_run_dir(args.run, args.logs_root)
  out = archive_train_run(
    run_dir,
    repo_root=args.repo_root.resolve(),
    archives_root=args.archives_root.resolve(),
    archive_id=args.archive_id,
  )
  print(f"[archive] wrote {out}")
  print(f"[archive] report  {out / 'report.md'}")
  print(f"[archive] manifest {out / 'manifest.json'}")
  print(f"[agent-review] pending {out / 'agent_review.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
