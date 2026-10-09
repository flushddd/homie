#!/usr/bin/env bash
# Launch HomieRL_mjlab training (paper G1 / mjlab).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-}"
EXTRA_ARGS=("$@")

CMD=(
  uv run train Mjlab-Homie-Flat-Unitree-G1
  --env.scene.num-envs "${NUM_ENVS}"
  --agent.logger tensorboard
  --agent.upload-model False
)

if [[ -n "${MAX_ITERS}" ]]; then
  CMD+=(--agent.max-iterations "${MAX_ITERS}")
fi

CMD+=("${EXTRA_ARGS[@]}")
echo "[launch_train] ${CMD[*]}"
exec "${CMD[@]}"
