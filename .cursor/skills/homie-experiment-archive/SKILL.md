---
name: homie-experiment-archive
description: >-
  Archive HomieRL_mjlab train and play sessions, then review the new
  agent_review.md: record user-observed bend, stand, and velocity tracking,
  write hypotheses only from the param diff, propose concrete changes, and
  wait for explicit confirmation before editing training code or starting a
  run. Use after play/train finishes, when agent_review.md is pending, or
  when the user asks to review an archive. Never invent historical train
  params or unobserved behavior; mark gaps as 未记录.
---

# Homie experiment archive (train + play)

## Hard rules

- Train configs: only `<run>/params/{env,agent}.yaml`
- Play: official `mjlab.scripts.play` does **not** dump params; record `play_meta.json` (argv, ckpt, viewer, num_envs)
- Link train params via parent `params/` or sha256 match; else `未记录`
- Optional `play_resolved/` must be labeled archive-time reconstruction
- Never overwrite checkpoints/logs; reports: 数据观察 / 原因推测 / 已验证结论

## Judgment loop

After a train or play archive exists (`agent_review.md` status `pending`), or when the user asks to review that session:

1. Read that archive's `agent_review.md`, `report.md`, and `diff_vs_previous.json`.
2. If 弯腰 / 站立 / 速度跟随 are still `未记录`, ask the user what they saw. Do not infer behavior from the checkpoint name or from current source.
3. Write 原因推测 in `agent_review.md` using only the recorded param diff and the user's observations. If either side is `未记录`, the matching hypothesis stays `未记录`.
4. Write 待确认改动 as concrete files and parameter values.
5. Set `status` to `awaiting_confirmation` and stop. Do not edit training code, rewards, env config, or launch training until the user explicitly agrees.
6. After they agree, make only the confirmed change. The next archive's 现象记录 is how that change gets checked; do not mark the previous suggestion verified before that observation exists.

## Commands

```bash
cd HomieRL_mjlab

# Train
python scripts/experiment_archive/archive_train_run.py \
  logs/rsl_rl/homie_g1/<run>

# Play (no sim)
python scripts/experiment_archive/archive_play_session.py \
  --task Mjlab-Homie-Flat-Unitree-G1-Phase2 \
  --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \
  --num-envs 1 --viewer viser

# Play + auto-archive on exit
python scripts/experiment_archive/play_with_archive.py \
  Mjlab-Homie-Flat-Unitree-G1-Phase2 \
  --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \
  --num-envs 1 --viewer viser
```

## Paths

- Rules: `HomieRL_mjlab/scripts/experiment_archive/RULES.md`
- Scripts: `HomieRL_mjlab/scripts/experiment_archive/`
- Archives: `HomieRL_mjlab/experiments/archives/`
