"""HomieRL_mjlab task package — registers paper HOMIE G1 flat tasks."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import homie_g1_flat_env_cfg
from .rl.him_runner import HomieOnPolicyRunner
from .rl_cfg import homie_g1_him_runner_cfg

# Phase-1: legs-only; waist PD-held; arms random-pose curriculum (Homie).
register_mjlab_task(
  task_id="Mjlab-Homie-Flat-Unitree-G1",
  # Phase-1 actor one-step: proprio+12 act+10 container = 90.
  env_cfg=homie_g1_flat_env_cfg(phase=1),
  play_env_cfg=homie_g1_flat_env_cfg(play=True, phase=1),
  rl_cfg=homie_g1_him_runner_cfg(phase=1),
  runner_cls=HomieOnPolicyRunner,
)

# Phase-2: 14-act (legs+roll/pitch; yaw held), waist grasp targets, teacher mean/KL.
# Resume a phase-1 checkpoint so ``learn()`` can clone it as teacher.
register_mjlab_task(
  task_id="Mjlab-Homie-Flat-Unitree-G1-Phase2",
  env_cfg=homie_g1_flat_env_cfg(phase=2),
  play_env_cfg=homie_g1_flat_env_cfg(play=True, phase=2),
  rl_cfg=homie_g1_him_runner_cfg(phase=2),
  runner_cls=HomieOnPolicyRunner,
)
