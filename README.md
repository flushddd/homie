# Homie BPTT-only branch

This branch contains **only SuperMimic-style BPTT training code**, kept separate from the HIM-PPO mainline.

## Included

- `HomieRL_mjlab/src/homierl_mjlab/rl/stage3_bptt/` — Stage-3 body+EE BPTT
- `HomieRL_mjlab/src/homierl_mjlab/rl/stage12_bptt/` — Stage-1/2 loco BPTT experiment
- `HomieRL_mjlab/scripts/train_stage3_bptt.py`
- `HomieRL_mjlab/scripts/train_stage12_bptt.py`
- `HomieRL_mjlab/scripts/play_stage3_bptt.py`
- `HomieRL_mjlab/scripts/play_stage3_oracle_subgoal.py`
- `.vscode/launch.json` — BPTT-related launch entries only

## Not included

- HIM / RSL-RL PPO trainers (`rl/him*`, phase-1/2/3 PPO tasks)
- Legacy `HomieRL/legged_gym` Isaac PPO stack
- Unrelated deploy / robot asset dumps

## Runtime note

BPTT modules still **import** Homie env / optional frozen phase-2 policy helpers from the full `HomieRL_mjlab` package when you train. Use this branch as the isolated BPTT algorithm surface; merge or check out beside a full Homie checkout for sim deps, or develop BPTT here and port back to `stage3-bptt-crate-avoid` / `main`.

## Quick checks (needs full package on `PYTHONPATH`)

```bash
.venv/bin/python scripts/train_stage12_bptt.py --check
.venv/bin/python scripts/train_stage3_bptt.py --check
```
