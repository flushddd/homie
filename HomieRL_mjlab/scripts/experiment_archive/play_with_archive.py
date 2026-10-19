#!/usr/bin/env python3
"""Wrap mjlab play: dump play_meta (+ optional resolved cfg), run play, then archive.

Does not modify checkpoint or train logs. Adds write-only files under
``experiments/play_sessions/`` and ``experiments/archives/``.

Usage (from HomieRL_mjlab)::

  python scripts/experiment_archive/play_with_archive.py \\
    Mjlab-Homie-Flat-Unitree-G1-Phase2 \\
    --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \\
    --num-envs 1 --viewer viser
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MJLAB = _HERE.parents[1]
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from archive_lib import (  # noqa: E402
  UNRECORDED,
  archive_play_session,
  write_play_meta,
  _dump_play_resolved_from_current_code,
)


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  # First token is task id (same as mjlab.scripts.play).
  if not argv or argv[0].startswith("-"):
    print(
      "Usage: play_with_archive.py <task_id> [play args...]",
      file=sys.stderr,
    )
    return 2
  task_id = argv[0]
  play_argv = argv[1:]

  # Parse a subset of play flags we care to record (tyro handles the rest).
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--checkpoint-file", type=str, default=None)
  pre.add_argument("--num-envs", type=int, default=None)
  pre.add_argument("--viewer", type=str, default=None)
  pre.add_argument("--device", type=str, default=None)
  pre.add_argument("--agent", type=str, default="trained")
  pre.add_argument(
    "--dump-play-resolved",
    action="store_true",
    help="Also dump play=True cfg from current code before launching",
  )
  pre.add_argument(
    "--skip-play",
    action="store_true",
    help="Only dump+archive; do not launch viewer/sim",
  )
  known, remaining = pre.parse_known_args(play_argv)

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  session_dir = _MJLAB / "experiments" / "play_sessions" / f"{stamp}__{task_id}"
  session_dir.mkdir(parents=True, exist_ok=False)

  ckpt = Path(known.checkpoint_file) if known.checkpoint_file else None
  if ckpt is not None and not ckpt.is_absolute():
    cand = (_MJLAB / ckpt).resolve()
    ckpt = cand if cand.is_file() else ckpt.expanduser()

  full_argv = ["mjlab.scripts.play", task_id, *play_argv]
  write_play_meta(
    session_dir / "play_meta.json",
    task_id=task_id,
    checkpoint_file=str(ckpt) if ckpt else None,
    num_envs=known.num_envs,
    viewer=known.viewer,
    device=known.device,
    agent_mode=known.agent,
    argv=full_argv,
    extra={"remaining_cli": remaining},
  )

  if known.dump_play_resolved:
    _dump_play_resolved_from_current_code(
      session_dir / "play_resolved",
      task_id=task_id,
      num_envs=known.num_envs,
    )

  exit_code = 0
  if not known.skip_play:
    # Launch stock mjlab play with remaining args (without our wrapper-only flags).
    import mjlab.tasks  # noqa: F401
    import homierl_mjlab  # noqa: F401
    import mjlab
    import tyro
    from mjlab.scripts.play import PlayConfig, run_play
    from mjlab.tasks.registry import list_tasks

    # Rebuild PlayConfig from task + remaining play args (strip wrapper flags).
    filtered: list[str] = []
    skip_next = False
    wrapper_flags = {"--dump-play-resolved", "--skip-play"}
    for a in play_argv:
      if skip_next:
        skip_next = False
        continue
      if a in wrapper_flags:
        continue
      filtered.append(a)

    args = tyro.cli(
      PlayConfig,
      args=filtered,
      default=PlayConfig(),
      prog=f"play_with_archive {task_id}",
      config=mjlab.TYRO_FLAGS,
    )
    try:
      run_play(task_id, args)
    except SystemExit as e:
      exit_code = int(e.code or 0)
    except Exception:  # noqa: BLE001
      raise
  else:
    print(f"[play_with_archive] skip-play; session={session_dir}")

  # Always archive after play returns (or after skip).
  repo_root = _MJLAB.parent
  archives_root = _MJLAB / "experiments" / "archives"
  logs_root = _MJLAB / "logs" / "rsl_rl" / "homie_g1"
  out = archive_play_session(
    repo_root=repo_root,
    archives_root=archives_root,
    logs_root=logs_root,
    task_id=task_id,
    checkpoint_file=ckpt,
    num_envs=known.num_envs,
    viewer=known.viewer,
    device=known.device,
    agent_mode=known.agent,
    argv=full_argv,
    play_session_dir=session_dir,
    dump_play_resolved=False,  # already in session if requested
  )
  print(f"[play_with_archive] archived {out}")
  print(f"[agent-review] pending {out / 'agent_review.md'}")
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
