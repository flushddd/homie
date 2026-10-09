# HomieRL_mjlab

Paper HOMIE (`g1`) training on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp).

This package is **independent** of the IsaacGym `HomieRL/` tree. It ports:

- Lower-body 12-DoF velocity tracking
- Height / squat command + tracking rewards
- Upper-body pose curriculum (policy does not control arms/waist)
- HIM-PPO with left-right symmetry loss (`L_sym`)

## Install

```bash
cd HomieRL_mjlab
uv sync
```

Requires an NVIDIA GPU for training (mjlab / MuJoCo Warp).

## Train

Cursor / VS Code: Run and Debug → **HomieRL mjlab: Train (4096)**，或复合配置 **Train + TensorBoard**。

Shell:

```bash
cd HomieRL_mjlab
./scripts/launch_train.sh
# 可选: NUM_ENVS=64 MAX_ITERS=5 ./scripts/launch_train.sh
./scripts/launch_tensorboard.sh   # http://localhost:6007
```

等价命令:

```bash
uv run train Mjlab-Homie-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.logger tensorboard
```

Smoke test:

```bash
uv run train Mjlab-Homie-Flat-Unitree-G1 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard \
  --agent.upload-model False
```

## Play

```bash
uv run play Mjlab-Homie-Flat-Unitree-G1 --checkpoint-file logs/rsl_rl/homie_g1/<run>/model_*.pt
```

## Notes

- Robot asset uses mjlab's Unitree G1 from `asset_zoo` (compatible sensors / collisions). HOMIE default pose and PD scales are applied on top.
- Obs / action dims differ slightly from IsaacGym `g1` (29 DoF MJCF vs 27 in the old URDF). Checkpoints are **not** interchangeable.
- Original IsaacGym code under `HomieRL/` is unchanged.
