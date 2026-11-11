# Stage-1/2 SuperMimic BPTT (experimental)

**Parallel experiment** — does **not** replace HIM-PPO phase-1/2, and does **not**
block Stage-3 (`rl/stage3_bptt/`).

## Goal

Train loco with the same SuperMimic-style loop as Stage-3:

`collect (env PPO rewards) → world model → critic TD → short-horizon BPTT actor`

Curriculum:

1. **Phase-1** — `Mjlab-Homie-Flat-Unitree-G1` (12 leg acts; waist PD-held)
2. **Phase-2** — `Mjlab-Homie-Flat-Unitree-G1-Phase2` (14 acts: legs + waist roll/pitch)

Promote phase-1 → phase-2 only after stand/walk metrics look healthy (manual
`--phase 2` or future auto-gate). Fresh replay when switching phase (obs/act dims change).

## Non-claims

- Checkpoints here are **not** drop-in teachers for Stage-3 HIM (`99-D` history /
  Gaussian policy). Stage-3 should keep using PPO phase-2 ckpts until an explicit
  remap / teacher adapter exists.
- Scaffold actor is a **flat MLP** on a compact state/obs vector — not full HIM
  history estimator. That can be swapped later without touching Stage-3.
- Imagination rewards are a **loco proxy** (tilt / soft survival); the rich
  Phase-1/2 `Episode_Reward/*` stack is applied on **real collect** (`w_loco`).

## Layout

| File | Role |
|------|------|
| `cfg.py` | phase, dims, SuperMimic replay sizing |
| `networks.py` | `LocoActor` (12 or 14) + reuse Stage-3 `WorldModel` / critic |
| `collector.py` | Phase-1/2 env collect; action = actor body |
| `trainer.py` | WM → critic → BPTT (+ optional phase-1 teacher BC in phase-2) |
| `checks.py` | offline graph / dim checks |
| `scripts/train_stage12_bptt.py` | CLI |

## Commands

```bash
# Offline
.venv/bin/python scripts/train_stage12_bptt.py --check
.venv/bin/python scripts/train_stage12_bptt.py --synthetic-smoke --device cpu

# Phase-1 smoke (GPU)
.venv/bin/python scripts/train_stage12_bptt.py \
  --phase 1 --num-envs 16 --iters 20 --warmup-collect 64 \
  --run-name stage12_bptt_p1

# Phase-2 (optional --phase1-checkpoint for teacher BC)
.venv/bin/python scripts/train_stage12_bptt.py \
  --phase 2 --phase1-checkpoint logs/.../model_XXXX.pt \
  --num-envs 16 --iters 20 --run-name stage12_bptt_p2
```

Logs default under `logs/rsl_rl/homie_g1/<stamp>_stage12_bptt_p{1,2}/`.
