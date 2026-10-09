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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage
from legged_gym.utils.curriculum import CurriculumScheduler
import matplotlib.pyplot as plt
import os
import csv
class HIMPPO:
    actor_critic: HIMActorCritic
    def __init__(self,
                 actor_critic,
                 use_flip = True,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 symmetry_scale=1e-3,
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.update_num = 0 
        self.elbow_curriculum = CurriculumScheduler(
            start_step = 6000,
            end_step = 16000,
            min_value = 0.0,
            max_value = 0.01
        )
        self.action_std_list = []  # 存储每个 epoch 的动作标准差
        plt.ion()  # 开启交互模式，支持实时更新图形
        self.fig, self.ax = plt.subplots()  # 创建图形对象
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Action Standard Deviation')
        self.ax.set_title('Action Variability (Std) Over Time')        
                # ===== keep stage1 locomotion (LOCK) =====
        self.teacher_actor_critic = None      # stage1 teacher
        self.keep_loco_weight = 2.0           # 建议 2~5（越大越锁）
        self.keep_loco_dims = 15              # 你说的 12腿+3腰=15
        self.freeze_std_dims = 15           # 只锁腿std；如果你也想锁腰就写15

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        obs_sym = self.flip_g1_actor_obs(obs)
        critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
        self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_sym.actions).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs)
        self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
        self.transition_sym.rewards = rewards.clone()
        self.transition_sym.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        self.update_num +=1
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_actor_sym_loss = 0
        mean_critic_sym_loss = 0
        mean_elbow_loss = 0
        mean_loss_keep = 0

        mean_action_variability = 0  
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                self.actor_critic.act(obs_batch)

                # elbow_shape_batch = self.get_elbow_angle(obs_batch)
                # print("elbow_shape_batch.shape",elbow_shape_batch.shape)
                # print("batch:",elbow_shape_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
            # ===== LOCK: keep locomotion close to stage1 teacher =====

                loss_keep = 0.0
                waist_mask = (obs_batch[:, 99] < 1e-3).float()
                waist_count = waist_mask.sum().clamp(min=1.0)
                if self.teacher_actor_critic is not None and self.keep_loco_weight > 0:
                    with torch.no_grad():
                        # teacher 用 mean（不要 sample），避免噪声干扰
                        _ = self.teacher_actor_critic.act(obs_batch)
                        teacher_mu = self.teacher_actor_critic.action_mean
                        teacher_sigma = self.teacher_actor_critic.action_std
                    k = self.keep_loco_dims
                    diff = (mu_batch[:, :k] - teacher_mu[:, :k])

                    mu_s = mu_batch[:,:k]
                    sig_s = sigma_batch[:,:k]
                    mu_t = teacher_mu[:,:k]
                    sig_t = teacher_sigma[:,:k]
                    kl = self._kl_diag_gauss(mu_s,sig_s,mu_t,sig_t)
                    loss_keep = ((kl).sum() +(diff*diff).sum())/ ( k)
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                if self.use_flip:
                    flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                    flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
                    estimator_update_obs_batch =  torch.cat((obs_batch, flipped_obs_batch), dim=0)
                    estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                else:
                    estimator_update_obs_batch = obs_batch
                    estimator_update_next_critic_obs_batch = next_critic_obs_batch
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    
                if self.use_flip:
                    flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                    actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(self.actor_critic.act_inference(flipped_obs_batch) - self.flip_g1_actions(self.actor_critic.act_inference(obs_batch))), dim=-1))
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(self.actor_critic.evaluate(flipped_critic_obs_batch) - self.actor_critic.evaluate(critic_obs_batch).detach()))
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + actor_sym_loss + critic_sym_loss + self.keep_loco_weight * loss_keep
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # ## elbow_angle_loss
                # elbow_weight = self.elbow_curriculum.value(self.update_num)
                # # print(elbow_weight)
                # mask = elbow_shape_batch.squeeze(1)!=0
                # elbow_angle = elbow_shape_batch.squeeze(1)[mask]
                # # print("elbow_angle",elbow_angle)
                # ideal_angle = 1.5 # ~ 85
                # elbow_loss = ((elbow_angle -ideal_angle )**2).mean()* elbow_weight*0.01
                # # print("elbow_loss",elbow_loss)
                # loss += elbow_loss


                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                                # ===== LOCK: freeze std for legs (reduce exploration) =====
                if hasattr(self.actor_critic, "std") and self.actor_critic.std.grad is not None:
                    self.actor_critic.std.grad[:self.freeze_std_dims].zero_()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                mean_loss_keep += loss_keep
                # mean_elbow_loss += elbow_loss
                if self.use_flip:
                    mean_actor_sym_loss += actor_sym_loss.item()
                    mean_critic_sym_loss += critic_sym_loss.item()
        mean_action_variability += sigma_batch.mean().item()  # 累加每个 batch 的平均标准差
        print("mean_action_variability",mean_action_variability)
        self.action_std_list.append(mean_action_variability / self.num_mini_batches)
        print(f"Epoch {self.update_num} - Action Variability (Std): {mean_action_variability / self.num_mini_batches:.4f}")

        save_dir = "./logs"
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(save_dir, "action_std_list1.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
    
         # 如果文件不存在，先写表头
            if not file_exists:
                writer.writerow(["epoch", "action_std"])
    
    # 每次训练更新追加一行
            writer.writerow([self.update_num, mean_action_variability / self.num_mini_batches])

                # 每次更新时实时绘制图形
        self.ax.clear()  # 清除当前图像
        self.ax.plot(self.action_std_list)  # 绘制标准差曲线
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Action Standard Deviation')
        self.ax.set_title('Action Variability (Std) Over Time')

        # 强制刷新图形
        plt.pause(0.1)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        mean_loss_keep /= num_updates
        # mean_elbow_loss /= num_updates
        if self.use_flip:
            mean_actor_sym_loss /= num_updates
            mean_critic_sym_loss /= num_updates
        self.storage.clear()

        if self.use_flip:
            return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss,mean_loss_keep#,mean_elbow_loss
        else:
            return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss, 0, 0
    
    def flip_g1_actor_obs(self, obs):
        proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]
        
        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist
        flipped_proprioceptive_obs[:, :, 23] =  -proprioceptive_obs[:, :, 23] # waist
        flipped_proprioceptive_obs[:, :, 24] =  proprioceptive_obs[:, :, 24] # waist

        # flipped_proprioceptive_obs[:, :, 25] =  proprioceptive_obs[:, :, 32] # left shoulder
        # flipped_proprioceptive_obs[:, :, 26] = -proprioceptive_obs[:, :, 33]
        # flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34]
        # flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35] # elbow
        # flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36] # wrist
        # flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 37]
        # flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 38]

        
        # flipped_proprioceptive_obs[:, :, 32] =  proprioceptive_obs[:, :, 25] # right shoulder
        # flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 26]
        # flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27]
        # flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28] # elbow
        # flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29] # wrist
        # flipped_proprioceptive_obs[:, :, 37] =  proprioceptive_obs[:, :, 30]
        # flipped_proprioceptive_obs[:, :, 38] = -proprioceptive_obs[:, :, 31]
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 10+29] =  proprioceptive_obs[:, :, 16+29] # lower
        flipped_proprioceptive_obs[:, :, 11+29] = -proprioceptive_obs[:, :, 17+29]
        flipped_proprioceptive_obs[:, :, 12+29] = -proprioceptive_obs[:, :, 18+29]
        flipped_proprioceptive_obs[:, :, 13+29] =  proprioceptive_obs[:, :, 19+29]
        flipped_proprioceptive_obs[:, :, 14+29] =  proprioceptive_obs[:, :, 20+29]
        flipped_proprioceptive_obs[:, :, 15+29] = -proprioceptive_obs[:, :, 21+29]
        flipped_proprioceptive_obs[:, :, 16+29] =  proprioceptive_obs[:, :, 10+29]
        flipped_proprioceptive_obs[:, :, 17+29] = -proprioceptive_obs[:, :, 11+29]
        flipped_proprioceptive_obs[:, :, 18+29] = -proprioceptive_obs[:, :, 12+29]
        flipped_proprioceptive_obs[:, :, 19+29] =  proprioceptive_obs[:, :, 13+29]
        flipped_proprioceptive_obs[:, :, 20+29] =  proprioceptive_obs[:, :, 14+29]
        flipped_proprioceptive_obs[:, :, 21+29] = -proprioceptive_obs[:, :, 15+29]
        
        flipped_proprioceptive_obs[:, :, 22+29] =  -proprioceptive_obs[:, :, 22+29] # waist
        flipped_proprioceptive_obs[:, :, 23+27] =  -proprioceptive_obs[:, :, 23+29] # waist
        flipped_proprioceptive_obs[:, :, 24+27] =  proprioceptive_obs[:, :, 24+29] # waist

        # flipped_proprioceptive_obs[:, :, 25+29] =  proprioceptive_obs[:, :, 32+29] # left shoulder
        # flipped_proprioceptive_obs[:, :, 26+29] = -proprioceptive_obs[:, :, 33+29]
        # flipped_proprioceptive_obs[:, :, 27+29] = -proprioceptive_obs[:, :, 34+29]
        # flipped_proprioceptive_obs[:, :, 28+29] =  proprioceptive_obs[:, :, 35+29] # elbow
        # flipped_proprioceptive_obs[:, :, 29+29] = -proprioceptive_obs[:, :, 36+29] # wrist
        # flipped_proprioceptive_obs[:, :, 30+29] =  proprioceptive_obs[:, :, 37+29]
        # flipped_proprioceptive_obs[:, :, 31+29] = -proprioceptive_obs[:, :, 38+29]

        
        # flipped_proprioceptive_obs[:, :, 32+29] =  proprioceptive_obs[:, :, 25+29] # right shoulder
        # flipped_proprioceptive_obs[:, :, 33+29] = -proprioceptive_obs[:, :, 26+29]
        # flipped_proprioceptive_obs[:, :, 34+29] = -proprioceptive_obs[:, :, 27+29]
        # flipped_proprioceptive_obs[:, :, 35+29] =  proprioceptive_obs[:, :, 28+29] # elbow
        # flipped_proprioceptive_obs[:, :, 36+29] = -proprioceptive_obs[:, :, 29+29] # wrist
        # flipped_proprioceptive_obs[:, :, 37+29] =  proprioceptive_obs[:, :, 30+29]
        # flipped_proprioceptive_obs[:, :, 38+29] = -proprioceptive_obs[:, :, 31+29]
        
        # joint target
        flipped_proprioceptive_obs[:, :, 10+58] =  proprioceptive_obs[:, :, 16+58] # lower
        flipped_proprioceptive_obs[:, :, 11+58] = -proprioceptive_obs[:, :, 17+58]
        flipped_proprioceptive_obs[:, :, 12+58] = -proprioceptive_obs[:, :, 18+58]
        flipped_proprioceptive_obs[:, :, 13+58] =  proprioceptive_obs[:, :, 19+58]
        flipped_proprioceptive_obs[:, :, 14+58] =  proprioceptive_obs[:, :, 20+58]
        flipped_proprioceptive_obs[:, :, 15+58] = -proprioceptive_obs[:, :, 21+58]
        flipped_proprioceptive_obs[:, :, 16+58] =  proprioceptive_obs[:, :, 10+58]
        flipped_proprioceptive_obs[:, :, 17+58] = -proprioceptive_obs[:, :, 11+58]
        flipped_proprioceptive_obs[:, :, 18+58] = -proprioceptive_obs[:, :, 12+58]
        flipped_proprioceptive_obs[:, :, 19+58] =  proprioceptive_obs[:, :, 13+58]
        flipped_proprioceptive_obs[:, :, 20+58] =  proprioceptive_obs[:, :, 14+58]
        flipped_proprioceptive_obs[:, :, 21+58] = -proprioceptive_obs[:, :, 15+58]

        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length).detach()                                                                                                                                                                                                                                             
    
    def flip_g1_critic_obs(self, critic_obs):
        proprioceptive_obs = torch.clone(critic_obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        flipped_proprioceptive_obs[:, :, 3] =  proprioceptive_obs[:, :, 3] # height command
        flipped_proprioceptive_obs[:, :, 4] = -proprioceptive_obs[:, :, 4] # base ang vel roll
        flipped_proprioceptive_obs[:, :, 5] =  proprioceptive_obs[:, :, 5] # base ang vel pitch
        flipped_proprioceptive_obs[:, :, 6] = -proprioceptive_obs[:, :, 6] # base ang vel yaw
        flipped_proprioceptive_obs[:, :, 7] =  proprioceptive_obs[:, :, 7] # projected gravity x
        flipped_proprioceptive_obs[:, :, 8] = -proprioceptive_obs[:, :, 8] # projected gravity y
        flipped_proprioceptive_obs[:, :, 9] =  proprioceptive_obs[:, :, 9] # projected gravity z
        
        # joint pos
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # lower
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17]
        flipped_proprioceptive_obs[:, :, 12] = -proprioceptive_obs[:, :, 18]
        flipped_proprioceptive_obs[:, :, 13] =  proprioceptive_obs[:, :, 19]
        flipped_proprioceptive_obs[:, :, 14] =  proprioceptive_obs[:, :, 20]
        flipped_proprioceptive_obs[:, :, 15] = -proprioceptive_obs[:, :, 21]
        flipped_proprioceptive_obs[:, :, 16] =  proprioceptive_obs[:, :, 10]
        flipped_proprioceptive_obs[:, :, 17] = -proprioceptive_obs[:, :, 11]
        flipped_proprioceptive_obs[:, :, 18] = -proprioceptive_obs[:, :, 12]
        flipped_proprioceptive_obs[:, :, 19] =  proprioceptive_obs[:, :, 13]
        flipped_proprioceptive_obs[:, :, 20] =  proprioceptive_obs[:, :, 14]
        flipped_proprioceptive_obs[:, :, 21] = -proprioceptive_obs[:, :, 15]
        
        flipped_proprioceptive_obs[:, :, 22] =  -proprioceptive_obs[:, :, 22] # waist
        flipped_proprioceptive_obs[:, :, 23] =  -proprioceptive_obs[:, :, 23] # waist
        flipped_proprioceptive_obs[:, :, 24] =  proprioceptive_obs[:, :, 24] # waist

        # flipped_proprioceptive_obs[:, :, 25] =  proprioceptive_obs[:, :, 32] # left shoulder
        # flipped_proprioceptive_obs[:, :, 26] = -proprioceptive_obs[:, :, 33]
        # flipped_proprioceptive_obs[:, :, 27] = -proprioceptive_obs[:, :, 34]
        # flipped_proprioceptive_obs[:, :, 28] =  proprioceptive_obs[:, :, 35] # elbow
        # flipped_proprioceptive_obs[:, :, 29] = -proprioceptive_obs[:, :, 36] # wrist
        # flipped_proprioceptive_obs[:, :, 30] =  proprioceptive_obs[:, :, 37]
        # flipped_proprioceptive_obs[:, :, 31] = -proprioceptive_obs[:, :, 38]

        
        # flipped_proprioceptive_obs[:, :, 32] =  proprioceptive_obs[:, :, 25] # right shoulder
        # flipped_proprioceptive_obs[:, :, 33] = -proprioceptive_obs[:, :, 26]
        # flipped_proprioceptive_obs[:, :, 34] = -proprioceptive_obs[:, :, 27]
        # flipped_proprioceptive_obs[:, :, 35] =  proprioceptive_obs[:, :, 28] # elbow
        # flipped_proprioceptive_obs[:, :, 36] = -proprioceptive_obs[:, :, 29] # wrist
        # flipped_proprioceptive_obs[:, :, 37] =  proprioceptive_obs[:, :, 30]
        # flipped_proprioceptive_obs[:, :, 38] = -proprioceptive_obs[:, :, 31]
        
        # joint vel
        flipped_proprioceptive_obs[:, :, 10+29] =  proprioceptive_obs[:, :, 16+29] # lower
        flipped_proprioceptive_obs[:, :, 11+29] = -proprioceptive_obs[:, :, 17+29]
        flipped_proprioceptive_obs[:, :, 12+29] = -proprioceptive_obs[:, :, 18+29]
        flipped_proprioceptive_obs[:, :, 13+29] =  proprioceptive_obs[:, :, 19+29]
        flipped_proprioceptive_obs[:, :, 14+29] =  proprioceptive_obs[:, :, 20+29]
        flipped_proprioceptive_obs[:, :, 15+29] = -proprioceptive_obs[:, :, 21+29]
        flipped_proprioceptive_obs[:, :, 16+29] =  proprioceptive_obs[:, :, 10+29]
        flipped_proprioceptive_obs[:, :, 17+29] = -proprioceptive_obs[:, :, 11+29]
        flipped_proprioceptive_obs[:, :, 18+29] = -proprioceptive_obs[:, :, 12+29]
        flipped_proprioceptive_obs[:, :, 19+29] =  proprioceptive_obs[:, :, 13+29]
        flipped_proprioceptive_obs[:, :, 20+29] =  proprioceptive_obs[:, :, 14+29]
        flipped_proprioceptive_obs[:, :, 21+29] = -proprioceptive_obs[:, :, 15+29]
        
        flipped_proprioceptive_obs[:, :, 22+29] =  -proprioceptive_obs[:, :, 22+29] # waist
        flipped_proprioceptive_obs[:, :, 23+29] =  -proprioceptive_obs[:, :, 23+29] # waist
        flipped_proprioceptive_obs[:, :, 24+29] =  proprioceptive_obs[:, :, 24+29] # waist
        
        # flipped_proprioceptive_obs[:, :, 25+29] =  proprioceptive_obs[:, :, 32+29] # left shoulder
        # flipped_proprioceptive_obs[:, :, 26+29] = -proprioceptive_obs[:, :, 33+29]
        # flipped_proprioceptive_obs[:, :, 27+29] = -proprioceptive_obs[:, :, 34+29]
        # flipped_proprioceptive_obs[:, :, 28+29] =  proprioceptive_obs[:, :, 35+29] # elbow
        # flipped_proprioceptive_obs[:, :, 29+29] = -proprioceptive_obs[:, :, 36+29] # wrist
        # flipped_proprioceptive_obs[:, :, 30+29] =  proprioceptive_obs[:, :, 37+29]
        # flipped_proprioceptive_obs[:, :, 31+29] = -proprioceptive_obs[:, :, 38+29]

        
        # flipped_proprioceptive_obs[:, :, 32+29] =  proprioceptive_obs[:, :, 25+29] # right shoulder
        # flipped_proprioceptive_obs[:, :, 33+29] = -proprioceptive_obs[:, :, 26+29]
        # flipped_proprioceptive_obs[:, :, 34+29] = -proprioceptive_obs[:, :, 27+29]
        # flipped_proprioceptive_obs[:, :, 35+29] =  proprioceptive_obs[:, :, 28+29] # elbow
        # flipped_proprioceptive_obs[:, :, 36+29] = -proprioceptive_obs[:, :, 29+29] # wrist
        # flipped_proprioceptive_obs[:, :, 37+29] =  proprioceptive_obs[:, :, 30+29]
        # flipped_proprioceptive_obs[:, :, 38+29] = -proprioceptive_obs[:, :, 31+29]
        
        # joint target
        flipped_proprioceptive_obs[:, :, 10+58] =  proprioceptive_obs[:, :, 16+58] # lower
        flipped_proprioceptive_obs[:, :, 11+58] = -proprioceptive_obs[:, :, 17+58]
        flipped_proprioceptive_obs[:, :, 12+58] = -proprioceptive_obs[:, :, 18+58]
        flipped_proprioceptive_obs[:, :, 13+58] =  proprioceptive_obs[:, :, 19+58]
        flipped_proprioceptive_obs[:, :, 14+58] =  proprioceptive_obs[:, :, 20+58]
        flipped_proprioceptive_obs[:, :, 15+58] = -proprioceptive_obs[:, :, 21+58]
        flipped_proprioceptive_obs[:, :, 16+58] =  proprioceptive_obs[:, :, 10+58]
        flipped_proprioceptive_obs[:, :, 17+58] = -proprioceptive_obs[:, :, 11+58]
        flipped_proprioceptive_obs[:, :, 18+58] = -proprioceptive_obs[:, :, 12+58]
        flipped_proprioceptive_obs[:, :, 19+58] =  proprioceptive_obs[:, :, 13+58]
        flipped_proprioceptive_obs[:, :, 20+58] =  proprioceptive_obs[:, :, 14+58]
        flipped_proprioceptive_obs[:, :, 21+58] = -proprioceptive_obs[:, :, 15+58]
        
        flipped_proprioceptive_obs[:, :, 22+70+1+1+1+1+11] =  proprioceptive_obs[:, :, 22+70+1+1+1+1+11] # base lin vel x
        flipped_proprioceptive_obs[:, :, 23+70+1+1+1+1+11] = -proprioceptive_obs[:, :, 23+70+1+1+1+1+11] # base lin vel y
        flipped_proprioceptive_obs[:, :, 24+70+1+1+1+1+11] =  proprioceptive_obs[:, :, 24+70+1+1+1+1+11] # base lin vel z

        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length).detach()
    
    def flip_g1_actions(self, actions):
        flipped_actions = torch.zeros_like(actions)
        flipped_actions[:,  0] =  actions[:, 6]        # 0 "left_hip_pitch_joint",
        flipped_actions[:,  1] = -actions[:, 7]        # 1 "left_hip_roll_joint",
        flipped_actions[:,  2] = -actions[:, 8]        # 2 "left_hip_yaw_joint",
        flipped_actions[:,  3] =  actions[:, 9]        # 3 "left_knee_joint",
        flipped_actions[:,  4] =  actions[:, 10]       # 4 "left_ankle_pitch_joint",
        flipped_actions[:,  5] = -actions[:, 11]       # 5 "left_ankle_roll_joint",
        flipped_actions[:,  6] =  actions[:, 0]        # 6 "right_hip_pitch_joint",
        flipped_actions[:,  7] = -actions[:, 1]        # 7 "right_hip_roll_joint",
        flipped_actions[:,  8] = -actions[:, 2]        # 8 "right_hip_yaw_joint",
        flipped_actions[:,  9] =  actions[:, 3]        # 9 "right_knee_joint",
        flipped_actions[:, 10] =  actions[:, 4]        # 10 "right_ankle_pitch_joint",
        flipped_actions[:, 11] = -actions[:, 5]        # 11 "right_ankle_roll_joint",
        return flipped_actions.detach()
    
    # def get_elbow_angle(self,obs):
    #     proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_critic_obs * self.actor_critic.critic_history_length])
    #     proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.critic_history_length, self.actor_critic.num_one_step_critic_obs)
    #     # print("proprioceptive_obs.shape",proprioceptive_obs.shape)
    #     elbow_shape_angle = proprioceptive_obs[:,:,97]
    #     # print(elbow_shape_angle)
    #     return elbow_shape_angle

    def set_teacher(self, teacher_actor_critic, keep_loco_weight=2.0, keep_loco_dims=15, freeze_std_dims=12):
        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()
        for p in self.teacher_actor_critic.parameters():
            p.requires_grad = False
        self.keep_loco_weight = keep_loco_weight
        self.keep_loco_dims = keep_loco_dims
        self.freeze_std_dims = freeze_std_dims

    def _kl_diag_gauss(self, mu_s, sig_s, mu_t, sig_t, eps=1e-8):
        """ KL( N(mu_s, sig_s^2) || N(mu_t, sig_t^2) ), 对角高斯，返回 (B,) """
    
        # 将 mu 和 sig 截断并分离，以防止它们用于反向传播时不必要的计算图
        mu_s = mu_s.detach()
        sig_s = sig_s.detach()
        mu_t = mu_t.detach()
        sig_t = sig_t.detach()
    
        var_s = sig_s * sig_s
        var_t = sig_t * sig_t
        kl = 0.5 * (
        (var_s + (mu_s - mu_t).pow(2)) / (var_t + eps)
        - 1.0
        + 2.0 * torch.log((sig_t + eps) / (sig_s + eps))
        )
    
        return kl.sum(dim=-1)  # (B,)
    
    def plot_action_variability(self):
        # 绘制训练过程中每个 epoch 的动作标准差变化
        plt.plot(self.action_std_list)
        plt.xlabel('Epoch')
        plt.ylabel('Action Standard Deviation')
        plt.title('Action Variability (Std) Over Time')
        plt.show()
