#!/usr/bin/env python3
"""Archive a play session (config record; no sim unless --dump-play-resolved needs imports).

Examples::

  # Match the current VS Code play launch (no sim)
  python scripts/experiment_archive/archive_play_session.py \\
    --task Mjlab-Homie-Flat-Unitree-G1-Phase2 \\
    --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \\
    --num-envs 1 --viewer viser

  # If you know which train run produced the ckpt:
  python scripts/experiment_archive/archive_play_session.py \\
    --task Mjlab-Homie-Flat-Unitree-G1-Phase2 \\
    --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \\
    --train-run logs/rsl_rl/homie_g1/2026-10-17_15-05-22_phase2 \\
    --num-envs 1 --viewer viser
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from archive_lib import archive_play_session  # noqa: E402


def _defaults() -> tuple[Path, Path, Path]:
  mjlab_root = Path(__file__).resolve().parents[2]
  repo_root = mjlab_root.parent
  logs_root = mjlab_root / "logs" / "rsl_rl" / "homie_g1"
  archives_root = mjlab_root / "experiments" / "archives"
  return repo_root, logs_root, archives_root


def main(argv: list[str] | None = None) -> int:
  repo_root, logs_root, archives_root = _defaults()
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", required=True, help="Task id passed to mjlab.scripts.play")
  p.add_argument(
    "--checkpoint-file",
    type=Path,
    default=None,
    help="Play --checkpoint-file path",
  )
  p.add_argument("--num-envs", type=int, default=None)
  p.add_argument("--viewer", type=str, default=None)
  p.add_argument("--device", type=str, default=None)
  p.add_argument(
    "--agent-mode",
    type=str,
    default="trained",
    choices=("trained", "zero", "random"),
  )
  p.add_argument(
    "--train-run",
    type=Path,
    default=None,
    help="Optional train run dir containing params/ (overrides auto hash match)",
  )
  p.add_argument(
    "--play-session-dir",
    type=Path,
    default=None,
    help="Optional dir written by play_with_archive.py (play_meta.json ± play_resolved/)",
  )
  p.add_argument(
    "--dump-play-resolved",
    action="store_true",
    help=(
      "Also dump play=True env/agent from *current* code into play_resolved/ "
      "(explicitly labeled; not historical train params)"
    ),
  )
  p.add_argument("--archives-root", type=Path, default=archives_root)
  p.add_argument("--repo-root", type=Path, default=repo_root)
  p.add_argument("--logs-root", type=Path, default=logs_root)
  p.add_argument("--archive-id", type=str, default=None)
  args = p.parse_args(argv)

  ckpt = args.checkpoint_file
  if ckpt is not None and not ckpt.is_absolute():
    # Resolve relative to HomieRL_mjlab cwd convention.
    cand = (Path(__file__).resolve().parents[2] / ckpt).resolve()
    if cand.is_file():
      ckpt = cand
    else:
      ckpt = ckpt.expanduser()

  reconstructed_argv = [
    "mjlab.scripts.play",
    args.task,
  ]
  if ckpt is not None:
    reconstructed_argv += ["--checkpoint-file", str(ckpt)]
  if args.num_envs is not None:
    reconstructed_argv += ["--num-envs", str(args.num_envs)]
  if args.viewer is not None:
    reconstructed_argv += ["--viewer", args.viewer]
  if args.device is not None:
    reconstructed_argv += ["--device", args.device]
  if args.agent_mode != "trained":
    reconstructed_argv += ["--agent", args.agent_mode]

  out = archive_play_session(
    repo_root=args.repo_root.resolve(),
    archives_root=args.archives_root.resolve(),
    logs_root=args.logs_root.resolve(),
    task_id=args.task,
    checkpoint_file=ckpt,
    num_envs=args.num_envs,
    viewer=args.viewer,
    device=args.device,
    agent_mode=args.agent_mode,
    argv=reconstructed_argv,
    train_run_dir=args.train_run.resolve() if args.train_run else None,
    play_session_dir=(
      args.play_session_dir.resolve() if args.play_session_dir else None
    ),
    archive_id=args.archive_id,
    dump_play_resolved=bool(args.dump_play_resolved),
  )
  print(f"[archive-play] wrote {out}")
  print(f"[archive-play] report {out / 'report.md'}")
  print(f"[agent-review] pending {out / 'agent_review.md'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
