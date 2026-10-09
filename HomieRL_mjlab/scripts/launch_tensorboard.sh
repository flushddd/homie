#!/usr/bin/env bash
# Launch TensorBoard for HomieRL_mjlab runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-6007}"
LOGDIR="${LOGDIR:-logs/rsl_rl/homie_g1}"

echo "[launch_tensorboard] http://localhost:${PORT}  logdir=${LOGDIR}"
exec uv run tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
