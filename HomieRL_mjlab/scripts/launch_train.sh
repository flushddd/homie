#!/usr/bin/env bash
# Launch HomieRL_mjlab training (paper G1 / mjlab).
#
# Usage:
#   ./scripts/launch_train.sh                          # phase-1
#   PHASE=2 CHECKPOINT=model_xxxx.pt ./scripts/launch_train.sh
#   PHASE=3 CHECKPOINT=model_22000.pt ./scripts/launch_train.sh
#   PHASE=3 LOAD_RUN=phase2 CHECKPOINT=model_27600.pt NUM_ENVS=2048 ./scripts/launch_train.sh
#
# Env:
#   PHASE       1|2|3 (default 1)
#   NUM_ENVS    parallel envs (default 4096; phase-3 default 2048 on 16GB GPUs)
#   MAX_ITERS   optional max training iterations
#   LOAD_RUN    run dir name under logs/rsl_rl/homie_g1/ to resume
#               (phase-2 default: phase1; phase-3 default: phase2)
#   CHECKPOINT  checkpoint file name or regex (e.g. model_22000.pt)
#               If set with PHASE>=2, enables --agent.resume True
#   EXTRA args after the script are forwarded to `uv run train`
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-1}"
MAX_ITERS="${MAX_ITERS:-}"
LOAD_RUN="${LOAD_RUN:-}"
CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS=("$@")

case "${PHASE}" in
  1)
    TASK="Mjlab-Homie-Flat-Unitree-G1"
    DEFAULT_LOAD_RUN=""
    DEFAULT_NUM_ENVS=4096
    ;;
  2)
    TASK="Mjlab-Homie-Flat-Unitree-G1-Phase2"
    DEFAULT_LOAD_RUN="phase1"
    DEFAULT_NUM_ENVS=4096
    ;;
  3)
    TASK="Mjlab-Homie-Flat-Unitree-G1-Phase3"
    DEFAULT_LOAD_RUN="phase2"
    # Body/arm + container contacts; 4096 OOMs on ~16GB (RTX 5060 Ti).
    DEFAULT_NUM_ENVS=2048
    ;;
  *)
    echo "[launch_train] PHASE must be 1, 2, or 3 (got '${PHASE}')" >&2
    exit 1
    ;;
esac

NUM_ENVS="${NUM_ENVS:-${DEFAULT_NUM_ENVS}}"

CMD=(
  uv run train "${TASK}"
  --env.scene.num-envs "${NUM_ENVS}"
  --agent.logger tensorboard
  --agent.upload-model False
)

if [[ -n "${MAX_ITERS}" ]]; then
  CMD+=(--agent.max-iterations "${MAX_ITERS}")
fi

# Phase-2/3: resume prior-phase weights as student (cloned to teacher in learn()).
if [[ "${PHASE}" -ge 2 ]]; then
  if [[ -z "${LOAD_RUN}" ]]; then
    LOAD_RUN="${DEFAULT_LOAD_RUN}"
  fi
  if [[ -n "${CHECKPOINT}" ]]; then
    CMD+=(
      --agent.resume True
      --agent.load-run "${LOAD_RUN}"
      --agent.load-checkpoint "${CHECKPOINT}"
    )
  else
    echo "[launch_train] WARN: PHASE=${PHASE} without CHECKPOINT;" \
      "starting from scratch (no teacher warm-start)." >&2
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")
echo "[launch_train] PHASE=${PHASE} TASK=${TASK} NUM_ENVS=${NUM_ENVS}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "[launch_train] resume LOAD_RUN=${LOAD_RUN} CHECKPOINT=${CHECKPOINT}"
fi
echo "[launch_train] ${CMD[*]}"
exec "${CMD[@]}"
