# Stage-3 Training Fixes — Implementation Report

Based on `README_Stage3_Training_Fixes.md`. Date: 2026-11-10.

## Code audit (vs REWARD_FRAMEWORK summary)

| Item | Actual code (before) | Action |
|------|----------------------|--------|
| Imagination `present` | Not passed → safe/margin always on | **Fixed**: store `present` in replay; gate imag like collect |
| Reward clip | Imag + critic clamped; collect raw | **Kept by design**; log `*_clip_frac`; critic clip optional via cfg |
| `palm_step_fn` | Always on after WM | **Kept**, ablatable (`use_palm_step`) |
| Imagination survival | `surv≡1` | Soft tilt survival + `w_stability` (no hard truncate) |
| Smooth | Hard clamp ±0.2 on Δa | Huber-ish (quad core + linear tails) |
| Target only near | `exp(-d/0.08)` | Added `w_approach` far term + `w_soft_success` |
| Obs when absent crate | Could still query geom | Mask SDF→0, clearance→d_max |
| Checkpoint schema | Silent load | `reward_schema` check on load |

## Files changed

- `rewards_stage3.py` — terms API, approach/soft-success/stability, Huber smooth
- `cfg.py` — new weights / flags
- `replay.py` — `present` field; windows return truncated + present
- `trainer.py` — imag present gate, soft surv, term logs, clip frac
- `collector.py` — present into replay, real term logs, reach metrics, obs mask
- `train_stage3_bptt.py` — TB `RewardReal/*`, `RewardImag/*`, reach metrics
- `checks.py` — present in synthetic BPTT graph

## Not done (explicitly deferred per doc)

- P1 fixed validation set comparing WM vs `palm_step_fn` vs real FK errors (needs offline eval script + GPU rollouts).
- Full arm-sphere FK in imagination (still palm probes); documented gap.
- Rate-limit history inside imagination state (still approx).
- Margin ablation experiment (`w_margin=0`) — **configurable**, not run here.
- Multi-seed eval / open-box routing curriculum redesign.

## Compatibility

- **Replay**: new `present` channel → old in-memory buffers invalid; always rebuild from collect.
- **Checkpoints**: `reward_schema=stage3_v2_present_approach_stability`. Legacy `None` still loads weights; restart recommended for fair comparison.
- Do **not** resume mid-NaN / pre-alignment actor into this reward layout expecting continuous curves.

## Commands run

```bash
.venv/bin/python scripts/train_stage3_bptt.py --check
.venv/bin/python scripts/train_stage3_bptt.py --synthetic-smoke --device cpu
```

Results:

- `--check`: passed (`present_gate` absent safe/margin=0; present safe≪0; actor_grad_sum>0).
- `--synthetic-smoke --device cpu`: passed; `imag/w_safe=0` with `imag_present_frac=0`; `actor_grad_norm≈1e-3`.

## What to watch next train

- `RewardReal/w_safe` vs `RewardImag/w_safe` (should both be ~0 when present=0)
- `Train/reach_success_frac`, `Train/clear_reach_success_frac`, `Train/mean_goal_distance`
- `Train/imag_reward_clip_frac`, `Train/critic_reward_clip_frac`
- `Loss/actor_grad_norm` > 0
- `Train/neg_clearance_frac` and `Train/ppo_step_reward` (loco health)
