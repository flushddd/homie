# HomieRL_mjlab 实验归档操作规则

适用范围：`HomieRL_mjlab` 训练 / play 归档。本文件只写**操作规则**；Skill 说明与执行脚本分开存放。

## 已核实的项目锚点（勿猜测）

| 用途 | 位置 |
|------|------|
| 训练入口 | `mjlab.scripts.train`（`uv run train <task>` / `scripts/launch_train.sh`） |
| Play 入口 | `mjlab.scripts.play`（包装：`scripts/experiment_archive/play_with_archive.py`） |
| 训练配置落盘 | `mjlab/scripts/train.py` → `dump_yaml(log_dir/params/{env,agent}.yaml)` |
| Play 配置落盘 | **官方 play 不写 params**；由本工具写 `play_meta.json`（及可选 `play_resolved/`） |
| Checkpoint / 日志 | `HomieRL_mjlab/logs/rsl_rl/homie_g1/<run>/` |
| 部署参考配置 | `MujocoDeploy/g1.yaml`（不得用来回填历史训练参数） |
| 任务注册 | `HomieRL_mjlab/src/homierl_mjlab/__init__.py` |

## 训练归档

1. 只读原始 run：复制 `params/`，写独立归档目录；不覆盖 run 内模型/日志。
2. 配置来源：仅该次 `params/env.yaml` + `params/agent.yaml`。
3. 缺失字段写「未记录」；不得用当前源码回填历史训练值。
4. 记录 seed、代码版本（归档时刻）、Kp/Kd、action_scale、默认关节角、关节名、timestep、decimation、PPO/HIM、奖励；带 source。
5. 与上一归档 diff；报告分 **数据观察 / 原因推测 / 已验证结论**。

```bash
cd HomieRL_mjlab
python scripts/experiment_archive/archive_train_run.py logs/rsl_rl/homie_g1/<run_dir>
```

## Play 归档（已实现）

1. **必须记录**：task、checkpoint 路径、sha256、num_envs、viewer、device、argv。
2. **训练 params 关联**：优先 checkpoint 父目录 `params/`；否则在 logs 下按**同名文件 sha256** 匹配；再否则「未记录」。可用 `--train-run` 手工指定（仍要求该目录真有 params）。
3. **禁止**用当前源码冒充该次训练的 Kp/奖励等。
4. **可选** `--dump-play-resolved`：用**当前**代码 `load_env_cfg(play=True)` 导出到 `play_resolved/`，并强制写 `PROVENANCE.json` 声明「归档时刻重建，非历史训练 dump」。
5. 不覆盖原始 checkpoint；报告同样三段论。

```bash
# 只归档、不启动仿真（适合补录正在/已经 play 过的 CLI）
python scripts/experiment_archive/archive_play_session.py \
  --task Mjlab-Homie-Flat-Unitree-G1-Phase2 \
  --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \
  --num-envs 1 --viewer viser

# Play 结束后自动归档（会启动仿真/viewer）
python scripts/experiment_archive/play_with_archive.py \
  Mjlab-Homie-Flat-Unitree-G1-Phase2 \
  --checkpoint-file logs/rsl_rl/homie_g1/phase2/model_18800.pt \
  --num-envs 1 --viewer viser
```

输出：`experiments/archives/<timestamp>__play__<ckpt_stem>/`

## 判断环（agent，不是脚本）

每次训练或 play 归档都会写出 `agent_review.md`（`status: pending`）。脚本只填参数差异。

Agent 按 Skill「Judgment loop」执行：

1. 向用户核对弯腰、站立、速度跟随；没观察到的保持「未记录」。
2. 原因推测只引用参数差异和用户现象。
3. 写出待确认改动后停止。用户明确同意前，不改训练代码、不启动训练。
4. 同意后的改动，要等下一次归档里的现象记录才能写成已验证。

## 尚未实现

1. 验证数据分析（TB / episode metrics）  
2. 训练退出后的自动触发（play 包装器已覆盖 play 退出归档）  

## 目录约定

- 规则：本文件  
- Skill：`.cursor/skills/homie-experiment-archive/SKILL.md`  
- 脚本：`scripts/experiment_archive/`  
- 归档产物：`experiments/archives/`（gitignore）  
- Play 会话草稿：`experiments/play_sessions/`（gitignore）  
