# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from homierl_mjlab.rl.him.him_estimator import HIMEstimator


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class HIMActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_one_step_obs,
                        num_one_step_critic_obs,
                        actor_history_length,
                        critic_history_length,
                        num_actions=19,
                        actor_hidden_dims=[512, 256, 256],
                        critic_hidden_dims=[512, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.actor_proprioceptive_obs_length = self.actor_history_length * self.num_one_step_obs
        self.critic_proprioceptive_obs_length = self.critic_history_length * self.num_one_step_critic_obs
        self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
        self.num_critic_height_points = self.num_critic_obs - self.critic_proprioceptive_obs_length
        self.actor_use_height = True if self.num_height_points > 0 else False
        self.num_actions = num_actions

        self.dynamic_latent_dim = 32
        self.terrain_latent_dim = 32
        if self.actor_use_height:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim + self.terrain_latent_dim
        else:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim
        mlp_input_dim_c = num_critic_obs

        # Estimator
        self.estimator = HIMEstimator(temporal_steps=self.actor_history_length, num_one_step_obs=self.num_one_step_obs, num_height_points=0, latent_dim=self.dynamic_latent_dim)
        
        # Terrain Encoder
        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(self.num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.terrain_latent_dim),
            )

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')
        if self.actor_use_height:
            print(f'Terrain Encoder: {self.terrain_encoder}')
        

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        # Match Homie / runner clip_actions; also caps NaN recovery.
        self.action_mean_clip = 100.0

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def _sanitize_action_mean(self, action_mean: torch.Tensor) -> torch.Tensor:
        """Replace non-finite values and clamp to keep Normal(loc) valid."""
        if not torch.isfinite(action_mean).all():
            # One-time warn per blow-up burst (throttle via attribute).
            if not getattr(self, "_warned_nonfinite_mean", False):
                n_bad = int((~torch.isfinite(action_mean)).sum().item())
                print(
                    f"[WARN] actor mean non-finite ({n_bad} elems); "
                    "nan_to_num + clamp applied"
                )
                self._warned_nonfinite_mean = True
            action_mean = torch.nan_to_num(
                action_mean, nan=0.0, posinf=self.action_mean_clip, neginf=-self.action_mean_clip
            )
        else:
            self._warned_nonfinite_mean = False
        return action_mean.clamp(-self.action_mean_clip, self.action_mean_clip)

    def _actor_forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vel, dynamic_latent = self.estimator(
                obs_history[:, 0 : self.actor_proprioceptive_obs_length]
            )
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(
                obs_history[
                    :, -(self.num_height_points + self.num_one_step_obs) :
                ]
            )
            actor_input = torch.cat(
                (
                    obs_history[
                        :,
                        -(self.num_height_points + self.num_one_step_obs) : -self.num_height_points,
                    ],
                    vel,
                    dynamic_latent,
                    terrain_latent,
                ),
                dim=-1,
            )
        else:
            actor_input = torch.cat(
                (obs_history[:, -self.num_one_step_obs :], vel, dynamic_latent),
                dim=-1,
            )
        # Estimator can emit NaN once obs blow up; scrub before the actor MLP.
        actor_input = torch.nan_to_num(actor_input, nan=0.0, posinf=1e2, neginf=-1e2)
        return self._sanitize_action_mean(self.actor(actor_input))

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _positive_std(self) -> torch.Tensor:
        """Finite, strictly positive noise (NaN bypasses ``clamp(min=...)``)."""
        std = torch.nan_to_num(self.std, nan=1e-2, posinf=1.0, neginf=1e-2)
        return std.clamp(min=1e-4, max=1.0)

    def project_std_(self) -> None:
        """Keep ``self.std`` trainable but always a valid Normal scale."""
        with torch.no_grad():
            self.std.data.copy_(
                torch.nan_to_num(self.std.data, nan=1e-2, posinf=1.0, neginf=1e-2).clamp(
                    min=1e-4, max=1.0
                )
            )

    def update_distribution(self, obs_history):
        action_mean = self._actor_forward(obs_history)
        std = self._positive_std().expand_as(action_mean)
        self.distribution = Normal(action_mean, std, validate_args=False)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        return self._actor_forward(obs_history)

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
    
    def update_estimator(self, obs_history, next_critic_obs, lr=None):
        return self.estimator.update(obs_history[:, 0:self.actor_proprioceptive_obs_length], next_critic_obs[:, 0:self.critic_proprioceptive_obs_length], lr)