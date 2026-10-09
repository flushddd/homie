"""Homie HIM-PPO runner compatible with mjlab ``train`` CLI."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from homierl_mjlab.rl.history_adapter import HomieHistoryAdapter
from homierl_mjlab.rl.him.him_actor_critic import HIMActorCritic
from homierl_mjlab.rl.him.him_ppo import HIMPPO


class HomieOnPolicyRunner:
  """HIM-PPO + symmetry runner used as mjlab ``runner_cls``."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ):
    del kwargs
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.writer = None
    self.logger_type = str(train_cfg.get("logger", "tensorboard")).lower()

    him_cfg = train_cfg.get("him", {})
    self.actor_history_length = int(him_cfg.get("actor_history_length", 6))
    self.critic_history_length = int(him_cfg.get("critic_history_length", 1))
    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 50))
    self.upload_model = bool(train_cfg.get("upload_model", False))

    self.adapter = HomieHistoryAdapter(
      env,
      actor_history_length=self.actor_history_length,
      critic_history_length=self.critic_history_length,
    )
    self.env = self.adapter

    policy_hidden = tuple(
      train_cfg.get("actor", {}).get("hidden_dims", (512, 256, 128))
    )
    critic_hidden = tuple(
      train_cfg.get("critic", {}).get("hidden_dims", (512, 256, 128))
    )
    init_noise = float(
      train_cfg.get("actor", {})
      .get("distribution_cfg", {})
      .get("init_std", 1.0)
    )

    actor_critic = HIMActorCritic(
      self.adapter.num_obs,
      self.adapter.num_privileged_obs,
      self.adapter.num_one_step_obs,
      self.adapter.num_one_step_privileged_obs,
      self.actor_history_length,
      self.critic_history_length,
      self.adapter.num_actions,
      actor_hidden_dims=list(policy_hidden),
      critic_hidden_dims=list(critic_hidden),
      init_noise_std=init_noise,
    ).to(self.device)

    alg_cfg = train_cfg.get("algorithm", {})
    self.alg = HIMPPO(
      actor_critic,
      use_flip=bool(him_cfg.get("use_flip", True)),
      num_learning_epochs=int(alg_cfg.get("num_learning_epochs", 5)),
      num_mini_batches=int(alg_cfg.get("num_mini_batches", 4)),
      clip_param=float(alg_cfg.get("clip_param", 0.2)),
      gamma=float(alg_cfg.get("gamma", 0.99)),
      lam=float(alg_cfg.get("lam", 0.95)),
      value_loss_coef=float(alg_cfg.get("value_loss_coef", 1.0)),
      entropy_coef=float(alg_cfg.get("entropy_coef", 0.01)),
      learning_rate=float(alg_cfg.get("learning_rate", 3e-4)),
      max_grad_norm=float(alg_cfg.get("max_grad_norm", 1.0)),
      use_clipped_value_loss=bool(alg_cfg.get("use_clipped_value_loss", True)),
      schedule=str(alg_cfg.get("schedule", "adaptive")),
      desired_kl=float(alg_cfg.get("desired_kl", 0.01)),
      device=self.device,
      symmetry_scale=float(him_cfg.get("symmetry_scale", 0.5)),
      max_learning_rate=float(him_cfg.get("max_learning_rate", 1e-3)),
      min_learning_rate=float(him_cfg.get("min_learning_rate", 1e-5)),
    )

    # Configure joint symmetry from live robot joint names when possible.
    try:
      robot = env.unwrapped.scene["robot"]
      joint_names = list(robot.joint_names)
      self.alg.set_joint_symmetry(joint_names)
      self.alg.num_dof = len(joint_names)
    except Exception as exc:  # noqa: BLE001
      print(f"[WARN] Could not bind joint symmetry map: {exc}")

    self.alg.init_storage(
      self.adapter.num_envs,
      self.num_steps_per_env,
      [self.adapter.num_obs],
      [self.adapter.num_privileged_obs],
      [self.adapter.num_actions],
    )

    self.tot_timesteps = 0
    self.tot_time = 0.0
    self.current_learning_iteration = 0
    self.git_status_repos: list[str] = []
    self._viser_server = None
    self._viser_scene = None
    self._viser_every = 2
    self._viser_step = 0

    self.adapter.reset()

  def _train_viser_enabled(self) -> bool:
    flag = os.environ.get("HOMIE_TRAIN_VISER", "").strip().lower()
    return flag in ("1", "true", "yes", "on")

  def _start_train_viser(self) -> None:
    """Open a live Viser window that mirrors env 0 during ``learn``."""
    import viser
    from mjlab.viewer.viser.scene import MjlabViserScene

    env = self.adapter.unwrapped
    sim = env.sim
    self._viser_server = viser.ViserServer(label="Homie train")
    self._viser_scene = MjlabViserScene(
      server=self._viser_server,
      mj_model=sim.mj_model,
      num_envs=env.num_envs,
      sim_model=sim.model,
      expanded_fields=sim.expanded_fields,
    )
    self._viser_scene.env_idx = 0
    cfg = env.cfg.viewer
    with self._viser_server.gui.add_folder("Commands"):
      env.command_manager.create_gui(
        self._viser_server,
        lambda: 0,
        on_change=getattr(self._viser_scene, "request_update", None),
      )
    with self._viser_server.gui.add_folder("Scene"):
      self._viser_scene.create_scene_gui(
        camera_distance=float(cfg.distance),
        camera_azimuth=float(cfg.azimuth),
        camera_elevation=float(cfg.elevation),
      )
    self._viser_every = max(1, int(os.environ.get("HOMIE_TRAIN_VISER_EVERY", "2")))
    self._viser_step = 0
    print(
      "[INFO] Train Viser ON — open the Viser URL in a browser "
      f"(sync every {self._viser_every} env steps, env 0)."
    )

  def _sync_train_viser(self) -> None:
    if self._viser_scene is None or self._viser_server is None:
      return
    self._viser_step += 1
    if self._viser_step % self._viser_every != 0:
      return
    sim = self.adapter.unwrapped.sim
    with self._viser_server.atomic():
      self._viser_scene.update(sim.data)
      self._viser_server.flush()

  def _stop_train_viser(self) -> None:
    if self._viser_server is not None:
      try:
        self._viser_server.stop()
      except Exception:  # noqa: BLE001
        pass
    self._viser_server = None
    self._viser_scene = None

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if self._train_viser_enabled():
      self._start_train_viser()

    if self.log_dir is not None and self.writer is None:
      if self.logger_type == "wandb":
        try:
          import wandb
          from rsl_rl.utils import WandbLogWriter

          self.writer = WandbLogWriter(log_dir=self.log_dir, flush_secs=10)
        except Exception:
          self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
          self.logger_type = "tensorboard"
      else:
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

    if init_at_random_ep_len:
      self.adapter.episode_length_buf = torch.randint_like(
        self.adapter.episode_length_buf,
        high=int(self.adapter.max_episode_length),
      )

    obs = self.adapter.get_observations().to(self.device)
    critic_obs = self.adapter.get_privileged_observations().to(self.device)
    self.alg.actor_critic.train()

    rewbuffer: deque = deque(maxlen=100)
    lenbuffer: deque = deque(maxlen=100)
    cur_reward_sum = torch.zeros(self.adapter.num_envs, device=self.device)
    cur_episode_length = torch.zeros(self.adapter.num_envs, device=self.device)
    ep_extras: list[dict] = []

    start_iter = self.current_learning_iteration
    tot_iter = start_iter + num_learning_iterations

    try:
      self._learn_loop(
        start_iter,
        tot_iter,
        obs,
        critic_obs,
        rewbuffer,
        lenbuffer,
        cur_reward_sum,
        cur_episode_length,
        ep_extras,
      )
    finally:
      self._stop_train_viser()

  def _print_iteration_log(
    self,
    *,
    it: int,
    tot_iter: int,
    start_iter: int,
    fps: int,
    collection_time: float,
    learn_time: float,
    iteration_time: float,
    mean_value_loss: float,
    mean_surrogate_loss: float,
    mean_estimation_loss: float,
    mean_loss_keep: float,
    mean_swap_loss: float,
    mean_actor_sym_loss: float,
    mean_critic_sym_loss: float,
    mean_std: float,
    mean_rew: float,
    mean_ep_len: float,
    ep_means: dict,
    num_learning_iterations: int,
    width: int = 80,
    pad: int = 35,
  ) -> None:
    """Homie-style console log: reward, episode length, collection/learn time."""
    title = f" \033[1m Learning iteration {it}/{tot_iter - 1} \033[0m "
    log_string = f"{'#' * width}\n{title.center(width, ' ')}\n\n"
    log_string += (
      f"{'Computation:':>{pad}} {fps:.0f} steps/s "
      f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
      f"{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"
      f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
      f"{'Estimation loss:':>{pad}} {mean_estimation_loss:.4f}\n"
      f"{'Keep loss:':>{pad}} {mean_loss_keep:.4f}\n"
      f"{'Swap loss:':>{pad}} {mean_swap_loss:.4f}\n"
      f"{'Mean actor sym loss:':>{pad}} {mean_actor_sym_loss:.4f}\n"
      f"{'Mean critic sym loss:':>{pad}} {mean_critic_sym_loss:.4f}\n"
      f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
    )
    if mean_rew == mean_rew:  # not NaN
      log_string += (
        f"{'Mean reward:':>{pad}} {mean_rew:.2f}\n"
        f"{'Mean episode length:':>{pad}} {mean_ep_len:.2f}\n"
      )
    for key, val in sorted(ep_means.items()):
      log_string += f"{f'Mean episode {key}:':>{pad}} {val:.4f}\n"
    done_iters = it - start_iter + 1
    remaining = tot_iter - it - 1
    eta = self.tot_time / max(done_iters, 1) * max(remaining, 0)
    del num_learning_iterations
    log_string += (
      f"{'-' * width}\n"
      f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
      f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
      f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
      f"{'ETA:':>{pad}} {eta:.1f}s\n"
    )
    print(log_string)

  def _learn_loop(
    self,
    start_iter,
    tot_iter,
    obs,
    critic_obs,
    rewbuffer,
    lenbuffer,
    cur_reward_sum,
    cur_episode_length,
    ep_extras,
  ):
    for it in range(start_iter, tot_iter):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.num_steps_per_env):
          actions = self.alg.act(obs, critic_obs)
          (
            obs,
            privileged_obs,
            rewards,
            dones,
            infos,
            termination_ids,
            termination_privileged_obs,
          ) = self.adapter.step(actions)

          critic_obs = privileged_obs
          obs = obs.to(self.device)
          critic_obs = critic_obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          termination_ids = termination_ids.to(self.device)
          termination_privileged_obs = termination_privileged_obs.to(self.device)

          next_critic_obs = critic_obs.clone()
          if len(termination_ids):
            next_critic_obs[termination_ids] = termination_privileged_obs

          self.alg.process_env_step(rewards, dones, infos, next_critic_obs)

          # mjlab puts per-term episode rewards in extras["log"] on reset
          # (keys like Episode_Reward/tracking_base_height).
          if isinstance(infos, dict):
            if "log" in infos and infos["log"]:
              ep_extras.append(infos["log"])
            elif "episode" in infos and infos["episode"]:
              ep_extras.append(infos["episode"])

          cur_reward_sum += rewards
          cur_episode_length += 1
          new_ids = (dones > 0).nonzero(as_tuple=False)
          rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
          lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
          cur_reward_sum[new_ids] = 0
          cur_episode_length[new_ids] = 0
          self._sync_train_viser()

        collection_time = time.time() - start
        start = time.time()
        self.alg.compute_returns(critic_obs)

      (
        mean_value_loss,
        mean_surrogate_loss,
        mean_estimation_loss,
        mean_swap_loss,
        mean_actor_sym_loss,
        mean_critic_sym_loss,
        mean_loss_keep,
      ) = self.alg.update()
      learn_time = time.time() - start

      self.current_learning_iteration = it
      self.tot_timesteps += self.num_steps_per_env * self.adapter.num_envs
      self.tot_time += collection_time + learn_time

      iteration_time = collection_time + learn_time
      fps = int(
        self.num_steps_per_env
        * self.adapter.num_envs
        / max(iteration_time, 1e-6)
      )
      mean_std = float(self.alg.actor_critic.std.mean().item())
      mean_rew = statistics.mean(rewbuffer) if rewbuffer else float("nan")
      mean_ep_len = statistics.mean(lenbuffer) if lenbuffer else float("nan")

      # Aggregate per-term episode rewards for TB + console (Homie-style).
      ep_means: dict[str, float] = {}
      if ep_extras:
        all_keys = {k for ep in ep_extras for k in ep}
        for key in all_keys:
          vals = []
          for ep in ep_extras:
            if key not in ep:
              continue
            v = ep[key]
            if isinstance(v, torch.Tensor):
              vals.append(float(v.detach().mean().cpu()))
            else:
              vals.append(float(v))
          if vals:
            ep_means[key] = sum(vals) / len(vals)
        ep_extras.clear()

      if self.writer is not None:
        self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
        self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
        self.writer.add_scalar("Loss/estimation", mean_estimation_loss, it)
        self.writer.add_scalar("Loss/swap", mean_swap_loss, it)
        self.writer.add_scalar("Loss/actor_sym", mean_actor_sym_loss, it)
        self.writer.add_scalar("Loss/critic_sym", mean_critic_sym_loss, it)
        self.writer.add_scalar("Loss/keep", mean_loss_keep, it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)
        if len(rewbuffer) > 0:
          self.writer.add_scalar("Train/mean_reward", mean_rew, it)
          self.writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
        for key, val in ep_means.items():
          self.writer.add_scalar(key, val, it)

      self._print_iteration_log(
        it=it,
        tot_iter=tot_iter,
        start_iter=start_iter,
        fps=fps,
        collection_time=collection_time,
        learn_time=learn_time,
        iteration_time=iteration_time,
        mean_value_loss=mean_value_loss,
        mean_surrogate_loss=mean_surrogate_loss,
        mean_estimation_loss=mean_estimation_loss,
        mean_loss_keep=mean_loss_keep,
        mean_swap_loss=mean_swap_loss,
        mean_actor_sym_loss=mean_actor_sym_loss,
        mean_critic_sym_loss=mean_critic_sym_loss,
        mean_std=mean_std,
        mean_rew=mean_rew,
        mean_ep_len=mean_ep_len,
        ep_means=ep_means,
        num_learning_iterations=tot_iter - start_iter,
      )

      if it % self.save_interval == 0 and self.log_dir is not None:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.adapter.unwrapped.common_step_counter}
    payload = {
      "model_state_dict": self.alg.actor_critic.state_dict(),
      "optimizer_state_dict": self.alg.optimizer.state_dict(),
      "estimator_optimizer_state_dict": self.alg.actor_critic.estimator.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": {**(infos or {}), "env_state": env_state},
    }
    torch.save(payload, path)

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location=None):
    del load_cfg
    loaded = torch.load(path, map_location=map_location or self.device, weights_only=False)
    self.alg.actor_critic.load_state_dict(loaded["model_state_dict"], strict=strict)
    if "optimizer_state_dict" in loaded:
      self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
    if "estimator_optimizer_state_dict" in loaded:
      self.alg.actor_critic.estimator.optimizer.load_state_dict(
        loaded["estimator_optimizer_state_dict"]
      )
    self.current_learning_iteration = int(loaded.get("iter", 0))
    infos = loaded.get("infos") or {}
    if "env_state" in infos:
      self.adapter.unwrapped.common_step_counter = infos["env_state"][
        "common_step_counter"
      ]
    return infos

  def add_git_repo_to_log(self, repo_file_path: str):
    self.git_status_repos.append(repo_file_path)

  def get_inference_policy(self, device: str | None = None):
    """Return a callable policy for mjlab ``play`` / viewers.

    mjlab viewers call ``policy(env.get_observations())`` with a TensorDict of
    one-step actor obs. HIM needs a stacked history, so this wrapper maintains
    that buffer and runs ``act_inference``.
    """
    if device is not None:
      self.alg.actor_critic.to(device)
    self.alg.actor_critic.eval()

    actor_critic = self.alg.actor_critic
    one_step = self.adapter.num_one_step_obs
    hist_len = self.actor_history_length
    run_device = device or self.device

    def _actor_step(obs) -> torch.Tensor:
      if isinstance(obs, TensorDict):
        step = obs["actor"]
        if isinstance(step, TensorDict):
          step = torch.cat([step[k] for k in step.keys(include_nested=False)], dim=-1)
      else:
        step = obs
      return step.to(run_device)

    class HomieInferencePolicy:
      def __init__(self):
        self._hist: torch.Tensor | None = None

      def reset(self):
        self._hist = None

      def __call__(self, obs) -> torch.Tensor:
        step = _actor_step(obs)
        if self._hist is None or self._hist.shape[0] != step.shape[0]:
          # Warm-start history with the current frame (same as adapter.reset).
          self._hist = step.repeat(1, hist_len)
        else:
          self._hist = torch.cat([self._hist[:, one_step:], step], dim=-1)
        with torch.inference_mode():
          return actor_critic.act_inference(self._hist)

    return HomieInferencePolicy()
