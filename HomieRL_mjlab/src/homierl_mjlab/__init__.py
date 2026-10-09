"""HomieRL_mjlab task package — registers paper HOMIE G1 flat task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import homie_g1_flat_env_cfg
from .rl.him_runner import HomieOnPolicyRunner
from .rl_cfg import homie_g1_him_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Homie-Flat-Unitree-G1",
  env_cfg=homie_g1_flat_env_cfg(),
  play_env_cfg=homie_g1_flat_env_cfg(play=True),
  rl_cfg=homie_g1_him_runner_cfg(),
  runner_cls=HomieOnPolicyRunner,
)
