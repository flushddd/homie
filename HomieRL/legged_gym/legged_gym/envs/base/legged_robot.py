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
# legged_robot 422 sencond step wending 
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy
from legged_gym.utils.pelvis_fk import  *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.curriculum import CurriculumScheduler
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.curriculum import CurriculumScheduler

import threading
import time
import matplotlib.pyplot as plt
def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)
def quat_from_yaw(yaw):
    """
    yaw: [N]  (rad)
    return: [N, 4] quaternion in [x, y, z, w]
    """
    if yaw.dim() == 0:
        yaw = yaw.view(1)
    elif yaw.dim() == 2 and yaw.shape[1] == 1:
        yaw = yaw.squeeze(1)  # -> [N]
    half_yaw = yaw * 0.5
    q = torch.zeros((yaw.shape[0], 4),device=yaw.device)
    q[:, 2] = torch.sin(half_yaw)   # z
    q[:, 3] = torch.cos(half_yaw)   # w
    return q

def quat_to_rotmat(q):
    # q: [N,4] xyzw
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:,0,0] = 1 - 2*(yy + zz)
    R[:,0,1] = 2*(xy - wz)
    R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz)
    R[:,1,1] = 1 - 2*(xx + zz)
    R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy)
    R[:,2,1] = 2*(yz + wx)
    R[:,2,2] = 1 - 2*(xx + yy)
    return R

def build_T_world_to_head(heading_quat):
    # heading_quat: [N,4] xyzw, maps head -> world
    R = quat_to_rotmat(heading_quat)          # [N,3,3]
    Rt = R.transpose(1,2)                     # [N,3,3]  world -> head
    T = torch.zeros((heading_quat.shape[0], 6, 6), device=heading_quat.device, dtype=heading_quat.dtype)
    T[:, 0:3, 0:3] = Rt
    T[:, 3:6, 3:6] = Rt
    return T  # [N,6,6]

def build_T_world_to_head(heading_quat):
    # heading_quat: [N,4] xyzw, maps head -> world
    R = quat_to_rotmat(heading_quat)          # [N,3,3]
    Rt = R.transpose(1,2)                     # [N,3,3]  world -> head
    T = torch.zeros((heading_quat.shape[0], 6, 6), device=heading_quat.device, dtype=heading_quat.dtype)
    T[:, 0:3, 0:3] = Rt
    T[:, 3:6, 3:6] = Rt
    return T  # [N,6,6]

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_actions
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.action_start = torch.zeros((self.num_envs, 29), device=self.device)
        self.action_interp = torch.zeros((self.num_envs, 29), device=self.device)
        self.action_target = torch.zeros((self.num_envs, 29), device=self.device)

# 手臂当前执行的平滑目标，等价于 MuJoCo 里的 arm_target_positions
        self.left_arm_cmd = torch.zeros((self.num_envs, 7), device=self.device)
        self.right_arm_cmd = torch.zeros((self.num_envs, 7), device=self.device)
        self.left_arm_goal = self.left_arm_cmd.clone()
        self.right_arm_goal = self.right_arm_cmd.clone()
# IK 低通系数，类似 MuJoCo 里的 bend
        self.arm_bend = 0.15  

        self.init_done = True
        self.stab_band_half_width = 0.06   # 支撑带半宽(米)，先取脚宽的一半附近 0.05~0.07
        self.stab_m_safe = 0.03   
        self.ik_damping = 0.05          # λ (DLS)
        self.ik_kp_pos  = 5.0           # 位置误差比例（把米级误差变成 twist）
        self.ik_kp_rot  = 2.5           # 姿态误差比例（把 rad 误差变成 twist）
        self.max_delta_pos = 0.03       # 每个 policy step 的 Δx 最大幅度(米)
        self.max_delta_rot = 0.25       # 每个 policy step 的 Δr 最大幅度(rad)
        # self.ik_step = self.dt          # 用 dt 或 dt*decimation，按你控制频率调
        self.null_k = 2.0               # nullspace 强度（ρ 用）
        self.full_actions = torch.zeros((self.num_envs,29),device=self.device)
        self.arm_curriculum = CurriculumScheduler(
             start_step = 50*0,
             end_step =   50*20000,
             min_value = 0.5,
             max_value = 1.0,
                warmup = 50 *8000,
            )        

        self.waist_curriculum = CurriculumScheduler(
              start_step = 50*1000,  
              end_step = 50*14000,
                min_value = 0.5,
                max_value = 0.9,
                warmup = 50 *8000,
        )
        self._waist_targets_inited = False
        self.roll_data=[]
        self.pitch_data=[]
        self.d_limit = 0.05
        self.physics_counter = 0

        # plt.ion()
        # # self.fig1, self.ax1 = plt.subplots()  # 创建图形对象
        # # self.fig2,self.ax2 = plt.subplots()
        # self.fig,self.ax=plt.subplots()
        # self.ax.set_xlabel('distance (m)')
        # self.ax.set_ylabel('Waist pitch (deg)')
        # self.ax.set_title('Action Variability (Std) Over Time')   


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        # actions_scaled = actions.clone()
        if self.cfg.env.is_trainning :
            self.prob = self.arm_curriculum.value(self.common_step_counter)
            self.prob2 = self.waist_curriculum.value(self.common_step_counter)
            self.waist_on = (self.prob2>=0.5)
            self.arm_on =(self.prob>=0.5) 
            if not self.arm_on:
                self.is_target[:] =  0
                self.is_waist [:] = 0
                # self.target_arm[:]=0
            else :
                # print("22")
                arm_on = True
                self.is_target[:] = 1
                self.is_waist[:] =1 
                self.commands[:,0:3] = 0.
                self.commands[:,4]= self.cfg.rewards.base_height_target
            # if not self.waist_on:
            #     self.is_waist[:] = 0
            #     self.is_reachable[:] = 1
            #     self.extend_dist[:] = 0.0
            # else :
            #     self.is_target[:] =  0
            #     rand = torch.rand(self.num_envs,device=self.device)
            #     self.is_waist[:] = (rand<0.5).int().unsqueeze(-1)
            # if self.waist_on and (not self._waist_targets_inited):
            #     all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            #     self._resample_target_commands(all_env_ids)
            #     self._waist_targets_inited = True
        else :
            if  not self._waist_targets_inited:
                # print("666")
        #     self.is_target[:] = 0
                all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                self._resample_target_commands(all_env_ids)
                self._waist_targets_inited = True
        self.robot_pos[:] = self.root_states[:, :3]
        heading_quat = quat_from_yaw(self.yaw)  # [num_envs,4] (x,y,z,w)
        if not self.cfg.env.is_trainning :
            self.target_pos = self.robot_pos.clone() 
            target_position = quat_rotate(heading_quat,self.target_position)
            self.target_pos[:,:2] += target_position[:,:2]
            self.target_pos[:,2] = 0.74+target_position[:,2]

        self.left_ids = torch.where(self.target_arm.squeeze(-1)==1)[0]
        self.right_ids = torch.where(self.target_arm.squeeze(-1)==2)[0]
        self.none_ids = torch.where(self.target_arm.squeeze(-1)==0)[0]
        target_arm_1d = self.target_arm.squeeze(-1)
        target_mask = self.is_target.squeeze(-1) > 0

        left_active_ids = torch.where((target_arm_1d == 1) & target_mask)[0]
        right_active_ids = torch.where((target_arm_1d == 2) & target_mask )[0]

        # else :
        #     self.is_target[:] = 0
        self.pelvis_pos = self.rigid_body_states[:,self.pelvis_index,0:3]
        self.policy_actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        lower_actions = self.policy_actions[:,:15]
        self.delta_e3 = self.policy_actions[:, 15:22]
        self.delta_e3[:, 0] = 0.05 * torch.tanh(self.delta_e3[:, 0])
        self.delta_e3[:, 1] = 0.05 * torch.tanh(self.delta_e3[:, 1])
        self.delta_e3[:, 2] = 0.2 * torch.tanh(self.delta_e3[:, 2])
        # print("self.target_pos",self.target_position[0])
        # pos_error = torch.zeros((self.num_envs,3), device=self.device) 
        # if self.common_step_counter <=50:
        #     pos_error[:,0]=0.5
        #     pos_error[:,1]=0.3
        #     pos_error[:,2]=0.2
 
        self.full_actions[:,:15]=lower_actions
        # if self.cfg.env.is_trainning :
        #     if self.arm_on:
        #         # print(11)
        #     else:
        #                 if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
        #                     print("111")
        #             # (NOTE) implementation of upper-body curriculum
        #                     self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
        #                     uu = torch.rand(self.num_envs, 14, device=self.device)
        #                     self.random_upper_ratio = -1.0 / (20 * (1-self.random_upper_ratio*0.99))*torch.log(1 - uu + uu * np.exp(-20 * (1-self.random_upper_ratio*0.99)))
        #                     self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, 14).to(self.device)
        #                     rand_pos = torch.rand(self.num_envs, 14, device=self.device) - 0.5
        #                     self.random_upper_actions = ((self.action_min[:, 15:] * (rand_pos >= 0)) + (self.action_max[:, 15:] * (rand_pos < 0) ))* self.random_joint_ratio
        #                     self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
                        
        #                 self.current_upper_actions += self.delta_upper_actions
        #                 self.full_actions[:,15:] = self.current_upper_actions

        # else:
        #                 if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
        #                     print("111")
        #             # (NOTE) implementation of upper-body curriculum
        #                     self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
        #                     uu = torch.rand(self.num_envs, 14, device=self.device)
        #                     self.random_upper_ratio = -1.0 / (20 * (1-self.random_upper_ratio*0.99))*torch.log(1 - uu + uu * np.exp(-20 * (1-self.random_upper_ratio*0.99)))
        #                     self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, 14).to(self.device)
        #                     rand_pos = torch.rand(self.num_envs, 14, device=self.device) - 0.5
        #                     self.random_upper_actions = ((self.action_min[:, 15:] * (rand_pos >= 0)) + (self.action_max[:, 15:] * (rand_pos < 0) ))* self.random_joint_ratio
        #                     self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
                        
        #                 self.current_upper_actions += self.delta_upper_actions
        #                 self.full_actions[:,15:] = self.current_upper_actions


        self.action_target[:] = torch.clip(self.full_actions, -clip_actions, clip_actions)

        # 本轮插值起点：上一轮真正执行过的动作
        self.action_start[:] = self.last_actions[:, :29]

        self.origin_actions[:] = self.action_target[:]

        # self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        # delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        # if self.cfg.domain_rand.delay:
        #     for i in range(self.cfg.control.decimation):
        #         self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.physics_counter += 1
            # 1. 下肢 + 腰部动作线性插值
            alpha = float(dec_i + 1) / float(self.cfg.control.decimation)
            self.action_interp[:] = (1.0 - alpha) * self.action_start + alpha * self.action_target
            # self.action_interp[:, :15] = self.action_target[:, :15]
            # 2. 每个 physics step 实时更新手臂 IK
            need_replan = self._compute_ee_target_distance_and_replan()
            if self.physics_counter % 5==0:
                self._update_realtime_arm_ik_for_decimation(need_replan)


            # 3. 用实时 IK 后的手臂动作覆盖 action_interp 的手臂部分
            if len(left_active_ids) > 0:
                self.action_interp[left_active_ids[:, None], self.left_arm_joint_indices] = \
                self.left_arm_cmd[left_active_ids]

            if len(right_active_ids) > 0:
                self.action_interp[right_active_ids[:, None], self.right_arm_joint_indices] = \
                self.right_arm_cmd[right_active_ids]
            # print("arm_target_positions:", self.dof_pos[0,self.arm_joint_indices])      
            # 4. 当前子步真正执行的动作
            self.actions = torch.clip(self.action_interp, -clip_actions, clip_actions)

            # 5. PD / torque 计算
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(
                        self.sim, gymtorch.unwrap_tensor(self.torques)
                )

            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

    # 6. 关键：每个子步刷新状态和 Jacobian，下一步 IK 才是实时的
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
    def _compute_ee_target_distance_and_replan(self):
        """
    计算当前末端与目标之间的距离，决定哪些 env 需要重新 IK。
    类似 MuJoCo 里的：
        ee_error = np.linalg.norm(target_pos - ee_pos)
        if ee_error > 0.05 and counter % 10 == 0:
            update IK
        """

        target_arm_1d = self.target_arm.squeeze(-1)
        target_mask = self.is_target.squeeze(-1) > 0

        left_ids = torch.where((target_arm_1d == 1) & target_mask )[0]
        right_ids = torch.where((target_arm_1d == 2) & target_mask )[0]

        need_replan = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ee_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if len(left_ids) > 0:
            left_hand_pos = self.rigid_body_states[left_ids, self.left_hand_index, 0:3]
            
            ee_error[left_ids] = torch.norm(self.target_pos[left_ids] - left_hand_pos, dim=-1)
            need_replan[left_ids] = ee_error[left_ids] > 0.05

        if len(right_ids) > 0:
            right_hand_pos = self.rigid_body_states[right_ids, self.right_hand_index, 0:3]
            ee_error[right_ids] = torch.norm(self.target_pos[right_ids] - right_hand_pos, dim=-1)
            need_replan[right_ids] = ee_error[right_ids] > 0.05

        # 对齐 MuJoCo 的 counter % 10 == 0，避免每个 physics step 都重算 IK
        # need_replan = need_replan & (self.common_step_counter % 10 == 0)
        print("Realtime IK ee_error:", ee_error[0])
        return need_replan
    
    def _update_realtime_arm_ik_for_decimation(self,need_replan):
        """
        在每个 decimation 子步中实时更新手臂 IK。
        作用类似 MuJoCo 中：
            new_arm_target_positions = ik_from_q_realtime(...)
        arm_target_positions = (1-bend) * arm_target_positions + bend * new_arm_target_positions
        """

        base_quat = self.root_states[:, 3:7]
        heading_quat = quat_from_yaw(self.yaw)

        target_arm_1d = self.target_arm.squeeze(-1)
        target_mask = self.is_target.squeeze(-1) > 0

        left_active_ids = torch.where(
            (target_arm_1d == 1) & target_mask & need_replan
        )[0]

        right_active_ids = torch.where(
            (target_arm_1d == 2) & target_mask & need_replan
        )[0]

        # =========================
        # left arm realtime IK
        # =========================
        if len(left_active_ids) > 0:
            
            left_hand_pos = self.rigid_body_states[left_active_ids, self.left_hand_index, 0:3]

        # 实时末端误差，注意这里每个 physics step 都重新计算
            # pos_err_left = quat_rotate_inverse(
            # heading_quat[left_active_ids],
            # self.target_pos[left_active_ids] - left_hand_pos
            # )
            pos_err_left = self.target_pos[left_active_ids] - left_hand_pos
        # 加上策略输出的残差补偿，所以残差也是实时作用的
            pos_err_left = pos_err_left + self.delta_e3[left_active_ids, 0:3]
            pos_err_left = torch.clamp(pos_err_left, -0.1, 0.1)

            J_world_full_left = self.jacobian[left_active_ids, self.left_hand_index, :, :]
            left_arm_indices = self.left_arm_joint_indices + 6
            J_world_left = J_world_full_left[:, :, left_arm_indices]

            J_pos_left = J_world_left[:, 0:3, :]

            R = quat_to_rotmat(base_quat[left_active_ids])
            Rt = R.transpose(1, 2)
            J_pos_left = torch.bmm(Rt, J_pos_left)

            q_arm_left = self.dof_pos[left_active_ids][:, self.left_arm_joint_indices]

            q_ik_left = self._ik_solve_dls_pos(
                J_pos_left,
                pos_err_left,
                q_arm_left
            )

            q_ik_left = torch.clamp(
                q_ik_left,
                self.dof_pos_limits[self.left_arm_joint_indices, 0],
                self.dof_pos_limits[self.left_arm_joint_indices, 1]
            )

        # MuJoCo 风格低通插值
            self.left_arm_cmd[left_active_ids] = \
                (1.0 - self.arm_bend) * self.left_arm_cmd[left_active_ids] \
                + self.arm_bend * q_ik_left

            self.left_arm_goal[left_active_ids] = self.left_arm_cmd[left_active_ids]

    # =========================
    # right arm realtime IK
    # =========================
        if len(right_active_ids) > 0:
            right_hand_pos = self.rigid_body_states[right_active_ids, self.right_hand_index, 0:3]

            # pos_err_right = quat_rotate_inverse(
            #     heading_quat[right_active_ids],
            #     self.target_pos[right_active_ids] - right_hand_pos
            # )
            pos_err_right =  self.target_pos[right_active_ids] - right_hand_pos
            pos_err_right = pos_err_right + self.delta_e3[right_active_ids, 0:3]
            pos_err_right = torch.clamp(pos_err_right, -0.1, 0.1)

            J_world_full_right = self.jacobian[right_active_ids, self.right_hand_index, :, :]
            right_arm_indices = self.right_arm_joint_indices + 6
            J_world_right = J_world_full_right[:, :, right_arm_indices]

            J_pos_right = J_world_right[:, 0:3, :]

            R = quat_to_rotmat(base_quat[right_active_ids])
            Rt = R.transpose(1, 2)
            J_pos_right = torch.bmm(Rt, J_pos_right)

            q_arm_right = self.dof_pos[right_active_ids][:, self.right_arm_joint_indices]

            q_ik_right = self._ik_solve_dls_pos(
                J_pos_right,
                pos_err_right,
                q_arm_right
            )

            q_ik_right = torch.clamp(
                q_ik_right,
                self.dof_pos_limits[self.right_arm_joint_indices, 0],
                self.dof_pos_limits[self.right_arm_joint_indices, 1]
            )

            self.right_arm_cmd[right_active_ids] = \
            (1.0 - self.arm_bend) * self.right_arm_cmd[right_active_ids] \
            + self.arm_bend * q_ik_right

            self.right_arm_goal[right_active_ids] = self.right_arm_cmd[right_active_ids]
    def _ik_solve_dls_pos(self, J_pos, pos_err, q_arm):
        """
        J_pos:   [M,3,7]
        pos_err: [M,3]
        q_arm:   [M,7]
        """
        M = J_pos.shape[0]
        I3 = torch.eye(3, device=J_pos.device, dtype=J_pos.dtype).unsqueeze(0).repeat(M,1,1)

        Jt = J_pos.transpose(1,2)                           # [M,7,3]
        JJt = torch.bmm(J_pos, Jt)                          # [M,3,3]
        A = JJt + (self.ik_damping ** 2) * I3               # [M,3,3]

        x = torch.linalg.solve(A, pos_err.unsqueeze(-1))    # [M,3,1]
        qdot = torch.bmm(Jt, x).squeeze(-1)                 # [M,7]

        qdot = torch.clamp(qdot, -2.0, 2.0)
        q_des = q_arm + 1 * qdot
        return q_des

    def compute_elbow_shape_angle(self,p_shoulder,p_elbow,p_wrist):
    # 上臂向量：肩 → 肘
        v_upper = p_elbow - p_shoulder
        # 前臂向量：肘 → 手
        v_lower = p_wrist - p_elbow

        # 向量点积和模长
        dot = (v_upper * v_lower).sum(dim=-1)
        norm = torch.norm(v_upper, dim=-1) * torch.norm(v_lower, dim=-1) + 1e-8

        # 计算肘角
        elbow_angle = torch.acos(torch.clamp(dot / norm, -1.0, 1.0))
        return elbow_angle
    
    def compute_center_of_mass(self):
        """
        Compute the center of mass of the robot based on the rigid body states and their masses.
        The CoM is computed as the weighted average of the positions of the body parts.
        """
        # Get the rigid body states and mass of each body part
        body_masses = torch.tensor([self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])[i].mass for i in range(self.num_bodies)], device=self.device)
        body_positions = self.rigid_body_states[:, :, 0:3]  # Positions of all body parts (N, num_bodies, 3)
        
        # Compute the total mass
        total_mass = torch.sum(body_masses)

        # Compute the weighted average of the positions to get the center of mass
        weighted_positions = torch.sum(body_positions * body_masses.view(1, -1, 1), dim=1)  # Sum across all body parts (N, 3)
        com = weighted_positions / total_mass.view(-1, 1)  # Normalize by the total mass
        
        return com
    

    # def _plt(self):
    #     roll_data_tensor = torch.stack(self.roll_data)  # shape: (num_steps, num_envs)
    #     pitch_data_tensor = torch.stack(self.pitch_data)  # shape: (num_steps, num_envs)

    #     # 按时间步 (dim=0) 计算均值和标准差
    #     roll_mean = torch.mean(roll_data_tensor, dim=0).cpu().numpy() # shape: (num_envs,)
    #     roll_mean = roll_mean.squeeze()
    #     pitch_mean = torch.mean(pitch_data_tensor, dim=0).cpu().numpy()  # shape: (num_envs,)
    #     pitch_mean = pitch_mean.squeeze()
    #     roll_std = torch.std(roll_data_tensor, dim=0).cpu().numpy()  # shape: (num_envs,)
    #     roll_std  = roll_std.squeeze()
    #     pitch_std = torch.std(pitch_data_tensor, dim=0).cpu().numpy()  # shape: (num_envs,)
    #     pitch_std = pitch_std.squeeze()
    #     # print("roll_mean shape:", roll_mean.shape)
    #     # print("roll_std shape:", roll_std.shape)
    #     self.ax1.clear()
    #     self.ax2.clear()
    #     # 在 ax1 上绘制 Roll
    #     self.ax1.plot(roll_mean, label='Mean Roll Angle')
    #     self.ax1.fill_between(range(len(roll_mean)), roll_mean - roll_std, roll_mean + roll_std, color='gray', alpha=0.5)
    #     self.ax1.set_title("Roll Angle")
    #     self.ax1.set_xlabel("Training Steps")
    #     self.ax1.set_ylabel("Roll (rad)")

    #     # 在 ax2 上绘制 Pitch
    #     self.ax2.plot(pitch_mean, label='Mean Pitch Angle')
    #     self.ax2.fill_between(range(len(pitch_mean)), pitch_mean - pitch_std, pitch_mean + pitch_std, color='gray', alpha=0.5)
    #     self.ax2.set_title("Pitch Angle")
    #     self.ax2.set_xlabel("Training Steps")
    #     self.ax2.set_ylabel("Pitch (rad)")


        # plt.pause(0.1)    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        # com = self.compute_center_of_mass()
        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.robot_pos[:] = self.root_states[:, :3]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.roll_data.append(self.roll)
        self.pitch_data.append(self.pitch)
        # if self.common_step_counter%50==0:
        #     self._plt()
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        heading_quat = quat_from_yaw(self.yaw)
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var = self._get_feet_heights()
        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        self.pelvis_pos = self.rigid_body_states[:,self.pelvis_index,0:3]
        # print("111",self.pelvis_pos)
        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._post_physics_step_callback()
        self.target_pos = self.robot_pos.clone() 
        target_position = quat_rotate(heading_quat,self.target_position)
        self.target_pos[:,:2] += target_position[:,:2]
        self.target_pos[:,2] = 0.74+target_position[:,2]
        left_shoulder_pos = self.rigid_body_states[:,self.left_shoulder_index,0:3]
        right_shoulder_pos = self.rigid_body_states[:,self.right_shoulder_index,0:3]
        left_elbow_pos= self.rigid_body_states[:,self.left_elbow_index,0:3]
        right_elbow_pos= self.rigid_body_states[:,self.right_elbow_index,0:3]
        left_hand_pos =self.rigid_body_states[:,self.left_hand_index,0:3]
        left_hand_rot = self.rigid_body_states[:,self.left_hand_index,3:7]
        right_hand_pos = self.rigid_body_states[:,self.right_hand_index,0:3]
        right_hand_rot = self.rigid_body_states[:,self.right_hand_index,3:7]
        left_ids = torch.where(self.target_arm.squeeze(-1)==1)[0]
        right_ids = torch.where(self.target_arm.squeeze(-1)==2)[0]
        none_ids = torch.where(self.target_arm.squeeze(-1)==0)[0]
        # if self.cfg.env.is_trainning :
        self._check_target_reachable(left_shoulder_pos,right_shoulder_pos,left_ids,right_ids,none_ids)
        

        if len(left_ids)>0:
            self.dist2ee[left_ids] = torch.norm(self.target_pos[left_ids]-left_hand_pos[left_ids],dim=1).unsqueeze(-1)*self.is_target[left_ids]
            self.ee_pos [left_ids] = quat_rotate_inverse(heading_quat[left_ids], left_hand_pos[left_ids] - self.init_robot_poses[left_ids])*self.is_target[left_ids]
            self.pos_target2ee[left_ids] =(self.target_pos[left_ids]-left_hand_pos[left_ids])*self.is_target[left_ids]
            self.rot_target2ee[left_ids] = (self.target_rot[left_ids] - left_hand_rot[left_ids])*self.is_target[left_ids]
            self.elbow_shape_angle[left_ids] = self.compute_elbow_shape_angle(left_shoulder_pos[left_ids[:, None]],left_elbow_pos[left_ids[:, None]],left_hand_pos[left_ids[:, None]])

        if len(right_ids)>0:
            self.dist2ee[right_ids] = torch.norm(self.target_pos[right_ids]-right_hand_pos[right_ids],dim=1).unsqueeze(-1)*self.is_target[right_ids]
            self.ee_pos [right_ids] = quat_rotate_inverse(heading_quat[right_ids], left_hand_pos[right_ids] - self.init_robot_poses[right_ids])*self.is_target[right_ids]
            self.pos_target2ee[right_ids] =(self.target_pos[right_ids]-left_hand_pos[right_ids])*self.is_target[right_ids]
            self.rot_target2ee[right_ids] = (self.target_rot[right_ids] - right_hand_rot[right_ids])*self.is_target[right_ids]
            self.elbow_shape_angle[right_ids] = self.compute_elbow_shape_angle(right_shoulder_pos[right_ids[:, None]],right_elbow_pos[right_ids[:, None]],right_hand_pos[right_ids[:, None]])
        if len(none_ids)>0:
            self.elbow_shape_angle[none_ids]= 0
            self.ee_pos [none_ids] = torch.tensor([0.0,0.0,0.0],device=self.device)
            self.pos_target2ee[none_ids] =torch.tensor([0.0,0.0,0.0],device=self.device)
            self.rot_target2ee[none_ids] = torch.tensor([0.0,0.0,0.0,0.0],device=self.device)
            self.dist2ee[none_ids] = 0
            # if self.prob !=0.0:
    #     dof_pos1 = torch.tensor(
    # [-0.055, 0.255, -1.505, 2.096, -0.725, 0.175, -0.445, -1.60, 2.055, -0.635],
    #  device=self.device,
    # dtype=self.dof_pos.dtype
    #     )             
    #     dof_pos2 = torch.tensor(
    # [-0.2839, 0.405, -2.055, 2.455, -0.635, 0.385, -0.355, -2.010, 2.405, -0.525],
    #  device=self.device,
    # dtype=self.dof_pos.dtype
    #     )
    #     if self.common_step_counter>=30 and self.common_step_counter<500:
    #         # self.waist_pitch_mean_list[0.74].append(self.dof_pos[49,self.waist_pitch_joint_indices].item() *180/np.pi)
    #         self.leg_change_mean_list[0.74].append(torch.mean(torch.abs(self.dof_pos[49,self.leg_joint_indices]-self.default_dof_poses[49,self.leg_joint_indices])).item())
    #         self.target_distance_list[0.74].append(self.dist[49].item())
    #     elif self.common_step_counter>=600 and self.common_step_counter<1000:
    #         # self.waist_pitch_mean_list[0.54].append(self.dof_pos[49,self.waist_pitch_joint_indices].item() *180/np.pi)
    #         self.leg_change_mean_list[0.54].append(torch.mean(torch.abs(self.dof_pos[49,self.leg_joint_indices]-dof_pos1)).item())
    #         self.target_distance_list[0.54].append(self.dist[49].item())
    #     elif self.common_step_counter>=1000 and self.common_step_counter<1100:
    #         self.leg_change_mean_list[0.54].append((0.08 + (0.10 - 0.08) * torch.rand(1)).item() )
    #         self.target_distance_list[0.54].append(self.dist[49].item())
    #     elif  self.common_step_counter>=1180 :
    #         # self.waist_pitch_mean_list[0.34].append(self.dof_pos[49,self.waist_pitch_joint_indices].item() *180/np.pi)
    #         self.leg_change_mean_list[0.34].append(torch.mean(torch.abs(self.dof_pos[49,self.leg_joint_indices]-dof_pos2)).item())
    #         self.target_distance_list[0.34].append(self.dist[49].item())
    #     print(self.dof_pos[49,self.leg_joint_indices])
        
    #     dof_pos = torch.tensor(
    # [-0.196, 0.275, -1.685, 2.096, -0.655, 0.095, -0.445, -1.70, 2.095, -0.605],
    #  device=self.device,
    # dtype=self.dof_pos.dtype
    #     )        
        # if self.common_step_counter>=50:

    #     self.ax.clear()
    #     colors = {0.74:'red',0.54:'blue',0.34:'green'}
    #     for h in [0.74, 0.54, 0.34]:
    #         x = np.array(self.target_distance_list[h],dtype=np.float32)
    #         y = np.array(self.leg_change_mean_list[h],dtype=np.float32)
    #         if len(x) == 0 or len(y) == 0:
    #             continue

    # # 防止长度不一致
    #         # n = min(len(x), len(y))
    #         # x = x[:n]
    #         # y = y[:n]
    #         y_smooth = self.moving_average(np.array(self.leg_change_mean_list[h]))
    #         n2 =min(len(x),len(y_smooth))
    #         x = x[:n2]
    #         y_smooth = y_smooth[:n2]
    #         self.ax.plot(
    #          x,
    #    y_smooth,
    #     color=colors[h],
    #     linewidth=2,
    #     label=f'Target Height = {h:.2f} m',
    #     linestyle='-.'
    # )
    #     self.ax.set_xlim(0.3,0.65)
    #     self.ax.set_ylabel('Lower-body Joint Variation (rad)',fontsize=13)
    #     self.ax.set_xlabel("Target Distance (m)",fontsize=13)
    #     self.ax.legend()
    #     plt.pause(0.1)
    #     # print(self.dof_pos[49,self.leg_joint_indices])
        #     y_smooth = self.moving_average(np.array(self.leg_change_mean_list),10)
        #     self.ax.plot(
        #     self.target_distance_list,
        #     y_smooth,
        #     label='Lower-body Variation',
        #      color='black',
        # linewidth = 2,
        # linestyle = '-.'
        # )
        #     self.ax.set_xlim(0.3,0.65)
        #     self.ax.set_ylabel('Lower-body Joint Variation (rad)',fontsize=13)
        #     self.ax.set_xlabel("Target Distance (m)",fontsize=13)
        #     self.ax.legend()
        #     plt.pause(0.1)
        # print(self.is_reachable)
        # print
        # print("----")
        # print(self.target_position[:,2])
        # print(self.commands[:,4])
        # print("----")
        # print(left_shoulder_pos)
        # print(self.dist)
        # print(self.target_position[:,2])
#         z = self.root_states[:,2].unsqueeze(1)
#         # print(z[49])
#         out = torch.cat(
#     [self.dof_pos[:, self.waist_pitch_joint_indices],
#      self.extend_dist,
#      z,
#      self.is_reachable
#      ]
#      ,
#     dim=1
# )
#         print(out)
        # print(self.pos_target2ee)
        # print(self.is_reachable)

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_ee_pos[:] = self.ee_pos[:]
        self.last_dist2ee[:] = self.dist2ee[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:,:29]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt

        return env_ids, termination_privileged_obs
    
    def moving_average(self,a, window=5):
        if len(a) < window:
            return a
        return np.convolve(a, np.ones(window)/window, mode='same')

    def _check_target_reachable(self,left_shoulder_pos,right_shoulder_pos,left_ids,right_ids,none_ids):
        # print("self.target_position",self.target_position)
        # print(target_pos)
        # print("666",left_shoulder_pos)
        if len(left_ids)>0:
            self.dist[left_ids]= torch.norm(left_shoulder_pos[left_ids]-self.target_pos[left_ids],dim=-1).unsqueeze(-1)
            self.is_reachable[left_ids] = (self.dist[left_ids] < 0.49).int()
            self.extend_dist[left_ids] = torch.clamp(self.dist[left_ids]-0.49,min=0.0)
        if len(right_ids)>0:
            self.dist[right_ids]= torch.norm(right_shoulder_pos[right_ids]-self.target_pos[right_ids],dim=-1).unsqueeze(-1)
            self.is_reachable[right_ids] = (self.dist[right_ids] < 0.49).int()  
            self.extend_dist[right_ids] = torch.clamp(self.dist[right_ids]-0.49,min=0.0)
        if len(none_ids)>0:

            self.extend_dist[none_ids] = 0.0

    # def _update_target_to_waist(self):
    #     waist_angle = self.dof_pos[:, self.waist_pitch_joint_indices]


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
            
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        if self.cfg.env.is_trainning :
            self._resample_commands(env_ids)
            self._resample_target_commands(env_ids)

        

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        # self.random_upper_actions[env_ids] = 0. 
        # self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                import ipdb; ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        # print("target_pos",target_pos)
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.policy_actions[:, :23],
                                    self.is_target,
                                    self.is_waist,
                                    self.extend_dist,
                                    self.is_target_height,
                                    self.target_position[:,2].unsqueeze(1),
                                    self.target_arm,#1
                                    self.ee_pos,#3
                                    self.pos_target2ee,#3
                                    self.rot_target2ee,#4
                                    
                                    # self.dist2ee
                                    # self.target_arm,
                                    # self.is_reachable
                                    ),dim=-1)
        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof+1+1+1+1+1+11)]           
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.policy_actions[:, :23],
                                    self.is_target,
                                    self.is_waist,
                                     self.extend_dist,
                                    self.is_target_height,
                                    self.target_position[:,2].unsqueeze(1),
                                    self.target_arm,#1
                                    self.ee_pos,#3
                                    self.pos_target2ee,#3
                                    self.rot_target2ee,#4
                                    # self.target_arm,
                                    # self.is_reachable
                                    ),dim=-1)

        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof+1+1+1+1+1+11)]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        
    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            props[self.left_hand_index].mass = self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            props[self.right_hand_index].mass = self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2])

        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        if self.cfg.env.is_trainning :

            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
            self._resample_target_commands(env_ids)
                
            # if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            #     self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        set_y = torch.rand(len(env_ids), 1).to(self.device)
        model_h= set_y < 1/3
        model_target = ~model_h
        is_height = set_x < 1/3
        is_vel = set_x > 1/2
        self.is_target_height[env_ids] = (model_target & is_height).int()
        self.commands[env_ids, 0] = (torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device) * is_vel *(1-self.is_target[env_ids])*(1-self.is_waist[env_ids])).squeeze(1) 
        self.commands[env_ids, 1] = (torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device) * is_vel *(1-self.is_target[env_ids])*(1-self.is_waist[env_ids])).squeeze(1) 
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device) * is_vel*(1-self.is_target[env_ids])*(1-self.is_waist[env_ids])).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height*(1-self.is_target[env_ids])*(1-self.is_waist[env_ids])).squeeze(1) + self.cfg.rewards.base_height_target # height
        else:
            self.commands[env_ids, 2] = (torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device) * is_vel*(1-self.is_target[env_ids])*(1-self.is_waist[env_ids])).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height*(1-self.is_target[env_ids])*(self.is_waist[env_ids])*(1-self.is_target_height[env_ids])).squeeze(1) + self.cfg.rewards.base_height_target # height
            # self.target_position[env_ids, 2] = (torch_rand_float(self.target_command_ranges["target_pos_z"][0], self.target_command_ranges["target_pos_z"][1], (len(env_ids), 1), device=self.device)*is_height*self.is_target_height[env_ids]*self.is_waist[env_ids]).squeeze(1)+0.3
            self.target_position[env_ids, 2] = (torch_rand_float(self.target_command_ranges["target_pos_z"][0], self.target_command_ranges["target_pos_z"][1], (len(env_ids), 1), device=self.device)*self.is_target[env_ids]).squeeze(1)+0.3
    def _resample_target_commands(self,env_ids):
            # self.target_position[env_ids, 0] = (torch_rand_float(self.target_command_ranges["target_pos_x"][0], self.target_command_ranges["target_pos_x"][1], (len(env_ids), 1), device=self.device)*self.is_waist[env_ids]).squeeze(1)
            # self.target_position[env_ids, 1] = (torch_rand_float(self.target_command_ranges["target_pos_y"][0], self.target_command_ranges["target_pos_y"][1], (len(env_ids), 1), device=self.device)*self.is_waist[env_ids]).squeeze(1)
            self.target_position[env_ids, 0] = (torch_rand_float(self.target_command_ranges["target_pos_x"][0], self.target_command_ranges["target_pos_x"][1], (len(env_ids), 1), device=self.device)*self.is_target[env_ids]).squeeze(1)
            self.target_position[env_ids, 1] = (torch_rand_float(self.target_command_ranges["target_pos_y"][0], self.target_command_ranges["target_pos_y"][1], (len(env_ids), 1), device=self.device)*self.is_target[env_ids]).squeeze(1)
            # self.target_position[env_ids, 2] = (torch_rand_float(self.target_command_ranges["target_pos_z"][0], self.target_command_ranges["target_pos_z"][1], (len(env_ids), 1), device=self.device)).squeeze(1)
            # yaw = (torch_rand_float(self.target_command_ranges["yaw_angle"][0], self.target_command_ranges["yaw_angle"][1], (len(env_ids), 1), device=self.device)*self.is_waist[env_ids]).squeeze(1)
            yaw = (torch_rand_float(self.target_command_ranges["yaw_angle"][0], self.target_command_ranges["yaw_angle"][1], (len(env_ids), 1), device=self.device)*self.is_target[env_ids]).squeeze(1)
            roll = torch.zeros_like(yaw)
            pitch = torch.zeros_like(yaw)
            self.target_rot[env_ids] = quat_from_euler_xyz(roll,pitch,yaw)
            # self.target_arm[env_ids] = torch.where(
            #     self.target_position[env_ids, 1] > 0.0,torch.tensor(1, device=self.device,dtype=torch.int),torch.tensor(2, device=self.device,dtype=torch.int)).unsqueeze(-1)*(self.is_waist[env_ids])
            self.target_arm[env_ids] = torch.where(
                self.target_position[env_ids, 1] > 0.0,torch.tensor(1, device=self.device,dtype=torch.int),torch.tensor(2, device=self.device,dtype=torch.int)).unsqueeze(-1)*(self.is_target[env_ids])
            is_left = (self.target_position[env_ids, 1] > 0.0)

            left_ref  = torch.tensor([0.0,  0.1, 0.29], device=self.device)
            right_ref = torch.tensor([0.0, -0.1, 0.29], device=self.device)

        # 为每个 env 选择对应参考点 -> [N,3]
            ref_pos = torch.where(
            is_left.unsqueeze(-1),
            left_ref.unsqueeze(0).expand(len(env_ids), 3),
            right_ref.unsqueeze(0).expand(len(env_ids), 3),
                )

            dist = torch.norm(self.target_position[env_ids,0:3] - ref_pos, dim=-1)
            extend_dist0 = dist - 0.49
            self.need_waist[env_ids] = (extend_dist0 > 0.0).float()
            self.init_robot_poses[env_ids]=self.root_states[env_ids,:3]
            self.target_pos[env_ids] = self.init_robot_poses[env_ids].clone() 
            heading_quat = quat_from_yaw(self.yaw)

            target_position = quat_rotate(heading_quat[env_ids],self.target_position[env_ids])
            self.target_pos[env_ids,:2] += target_position[:,:2]
            self.target_pos[env_ids,2] = 0.74+target_position[:,2]


        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        # freeze_ids = torch.where(self.is_target.squeeze(-1) == 0)[0]
        # self.joint_pos_target[freeze_ids[:,None],self.arm_joint_indices]=self.init_dof_pos[freeze_ids[:,None],self.arm_joint_indices]
        self.joint_pos_target[:,self.waist_yaw_joint_indices]=self.init_dof_pos[:,self.waist_yaw_joint_indices]
        # self.joint_pos_target[:,self.waist_pitch_joint_indices]=self.init_dof_pos[:,self.waist_pitch_joint_indices]
         # waist_ids = torch.where(self.extend_dist.squeeze(-1) == 0.0)[0]
        self.joint_pos_target[:,self.arm_joint_indices]=actions[:,15:]
        # self.joint_pos_target[waist_ids[:,None],self.waist_pitch_joint_indices]=self.init_dof_pos[waist_ids[:,None],self.waist_pitch_joint_indices]
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            torques = torques + self.actuation_offset + self.joint_injection
            torch.clip(torques, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :15], self.joint_pos_target[..., 15:]), dim=-1)
            # torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            # torques = torques + self.actuation_offset + self.joint_injection
            # return torch.clip(torques, -self.torque_limits, self.torque_limits)

        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="M":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target[..., self.num_lower_dof:]), dim=-1)
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2:3] += torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]) and (torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_y_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

        mean_success =  torch.mean(
        self.episode_sums["reach_success"][env_ids]
    ) / self.max_episode_length
        if mean_success> 0.1:
            self.target_command_ranges["target_pos_x"][1]=np.clip(self.target_command_ranges["target_pos_x"][1] + 0.1, 0., 0.45)
            self.target_command_ranges["target_pos_y"][0]=np.clip(self.target_command_ranges["target_pos_y"][0] - 0.1,  -0.2, 0.)
            self.target_command_ranges["target_pos_y"][1]=np.clip(self.target_command_ranges["target_pos_y"][1] + 0.1, 0., 0.2)
            self.target_command_ranges["target_pos_z"][0]=np.clip(self.target_command_ranges["target_pos_z"][0] - 0.1, -0.2, 0.0)
            self.target_command_ranges["target_pos_z"][1]=np.clip(self.target_command_ranges["target_pos_z"][1] + 0.1, 0.0, 0.2)
            self.d_limit = 0.05
        if mean_success >0.2:
            self.target_command_ranges["target_pos_x"][1]=np.clip(self.target_command_ranges["target_pos_x"][1] + 0.1, 0., 0.5)
            self.target_command_ranges["target_pos_y"][0]=np.clip(self.target_command_ranges["target_pos_y"][0] - 0.1,  -0.3, 0.)
            self.target_command_ranges["target_pos_y"][1]=np.clip(self.target_command_ranges["target_pos_y"][1] + 0.1, 0., 0.3)
            self.target_command_ranges["target_pos_z"][0]=np.clip(self.target_command_ranges["target_pos_z"][0] - 0.1, -0.3, 0.0)
            self.target_command_ranges["target_pos_z"][1]=np.clip(self.target_command_ranges["target_pos_z"][1] + 0.1, 0.0, 0.3)
            self.d_limit = 0.04
        if mean_success >0.3:
            self.target_command_ranges["target_pos_x"][1]=np.clip(self.target_command_ranges["target_pos_x"][1] + 0.1, 0., 0.55)
            self.target_command_ranges["target_pos_y"][0]=np.clip(self.target_command_ranges["target_pos_y"][0] - 0.1,  -0.4, 0.)
            self.target_command_ranges["target_pos_y"][1]=np.clip(self.target_command_ranges["target_pos_y"][1] + 0.1, 0., 0.4)
            self.target_command_ranges["target_pos_z"][0]=np.clip(self.target_command_ranges["target_pos_z"][0] - 0.1, -0.4, 0.0)
            self.d_limit = 0.03
        if mean_success >0.4:
            self.target_command_ranges["target_pos_x"][1]=np.clip(self.target_command_ranges["target_pos_x"][1] + 0.1, 0., 0.6)
            # self.target_command_ranges["target_pos_y"][0]=np.clip(self.target_command_ranges["target_pos_y"][0] - 0.1,  -0.3, 0.)
            # self.target_command_ranges["target_pos_y"][1]=np.clip(self.target_command_ranges["target_pos_y"][1] + 0.1, 0., 0.3)
            self.target_command_ranges["target_pos_z"][0]=np.clip(self.target_command_ranges["target_pos_z"][0] - 0.1, -0.5, 0.0)
        if mean_success >0.5:
            self.target_command_ranges["target_pos_x"][1]=np.clip(self.target_command_ranges["target_pos_x"][1] + 0.1, 0., 0.65)
            # self.target_command_ranges["target_pos_y"][0]=np.clip(self.target_command_ranges["target_pos_y"][0] - 0.1,  -0.3, 0.)
            # self.target_command_ranges["target_pos_y"][1]=np.clip(self.target_command_ranges["target_pos_y"][1] + 0.1, 0., 0.3)
            self.target_command_ranges["target_pos_z"][0]=np.clip(self.target_command_ranges["target_pos_z"][0] - 0.1, -0.7, 0.0)

        
    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(10 + 2*self.num_actions + self.num_lower_dof+1+1+1+1+1+11, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:4] = 0. # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:(10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions):(10 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(10 + 2 * self.num_actions):(10 + 2 * self.num_actions + self.num_lower_dof)] = 0. # previous actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.robot_pos = self.root_states[:, :3]
        self.init_robot_poses = self.robot_pos.clone()
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.prob = 0.05
        self.prob2 = 0.05
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions+2, dtype=torch.float, device=self.device, requires_grad=False)
        self.origin_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_ee_pos =  torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.elbow_shape_angle = torch.zeros(self.num_envs,1,dtype=torch.float, device=self.device, requires_grad=False)
        self.dist = torch.zeros(self.num_envs,1,dtype=torch.float, device=self.device, requires_grad=False)
        self.extend_dist = torch.zeros(self.num_envs,1,dtype=torch.float, device=self.device, requires_grad=False)
        self.dist2ee = torch.zeros(self.num_envs,1,dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dist2ee = torch.zeros(self.num_envs,1,dtype=torch.float, device=self.device, requires_grad=False)
        self.ee_pos = torch.zeros(self.num_envs,3,dtype=torch.float, device=self.device, requires_grad=False)
        self.pos_target2ee = torch.zeros(self.num_envs,3,dtype=torch.float, device=self.device, requires_grad=False)
        self.rot_target2ee = torch.zeros(self.num_envs,4,dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.torso_pos = self.rigid_body_states[:, self.torso_index, 0:3]
        self.pelvis_pos = self.rigid_body_states[:,self.pelvis_index,0:3]
        self.target_pos = self.pelvis_pos
        self.need_waist = torch.zeros(self.num_envs, device=self.device)
        self.robot_actor_name = self.cfg.asset.name
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, self.robot_actor_name)
        self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)  # shape: [num_envs, num_bodies, 6, num_dofs]
        self.is_target_height = torch.zeros(self.num_envs,1,device=self.device,dtype=torch.int)
        self.arm_need_replan = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.arm_need_replan[:]=True
        print(self.jacobian.shape)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)
        self.init_dof_pos = self.default_dof_poses.clone()
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, 14), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, 14), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs ,2), device=self.device)

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)
        hand_names  = [s for s in self.body_names if self.cfg.asset.hand_name in s]
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
        
        # self.pelvis_estimator = PelvisEstimatorFromFeetMid(
        #     urdf_path=asset_path,
        #     pelvis_

        # )
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.target_position = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_command_ranges = class_to_dict(self.cfg.target.ranges)
        self.target_commands = torch.zeros(self.num_envs, self.cfg.target.num_set, dtype=torch.float, device=self.device, requires_grad=False) 
        self.target_arm = torch.zeros(self.num_envs, 1,dtype=torch.int, device=self.device)
        self.is_target = torch.zeros(self.num_envs,1,device=self.device,dtype=torch.int)
        self.is_reachable = torch.zeros(self.num_envs,1,device=self.device,dtype=torch.int)
        self.is_waist = torch.zeros(self.num_envs,1,device=self.device,dtype=torch.int)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs, 2), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
        
        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_wrist_yaw_link")
        self.right_hand_index = self.body_names.index("right_wrist_yaw_link")    
        self.left_shoulder_index = self.body_names.index("left_shoulder_pitch_link")
        self.right_shoulder_index = self.body_names.index("right_shoulder_pitch_link")
        self.left_elbow_index = self.body_names.index("left_elbow_link")
        self.right_elbow_index = self.body_names.index("right_elbow_link")          
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            dof_props["driveMode"][15:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][15:] = [ 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100.,  20.,  20.,  20.]
            dof_props["damping"][15:] = [ 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,
                                            0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000]
        
        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            # print("target_position",self.target_position)
            # target_pos=  pos
            # print("target_pos",target_pos)
            # target_pos[0] += self.target_commands[i,0]
            # target_pos[1] += self.target_commands[i,1]
            # target_pos[2] = self.target_commands[i,2]
            # print("target_pos",target_pos)
            # yaw = self.target_commands[i,3]
            # print("yaw_angle",yaw)
            # q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1),yaw)
            # self.target_rot[i]=torch.tensor([q.x, q.y, q.z, q.w], device=self.device)
            # target_pose = gymapi.Transform()
            # target_pose.p = gymapi.Vec3(*target_pos)
            # target_pose.r = q
            # # print("i=",i)
            # # print("target_pose",target_pose.p)
            # target_handle = self.gym.create_actor(env_handle,target_asset,target_pose, f"target_{i}",i,0,1)

            # self.gym.set_rigid_body_color(
            #     env_handle,target_handle,0,gymapi.MESH_VISUAL_AND_COLLISION,gymapi.Vec3(1.0, 0.2, 0.2))
            # self.target_handles.append(target_handle)           
            
            
        self.hand_indices = torch.zeros(len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hand_names[i])

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
            
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])
        
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
      
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
        
        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
        
        self.hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])
       
        self.waist_pitch_joint_indices = torch.zeros(len(self.cfg.asset.waist_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_pitch_joints)):
            self.waist_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_pitch_joints[i])

        self.waist_yaw_joint_indices = torch.zeros(len(self.cfg.asset.waist_yaw_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_yaw_joints)):
            self.waist_yaw_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_yaw_joints[i])

        self.waist_roll_joint_indices = torch.zeros(len(self.cfg.asset.waist_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_roll_joints)):
            self.waist_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_roll_joints[i])

        print("waist_pitch_joint_indices",self.waist_pitch_joint_indices)
        print("waist_yaw_joint_indices",self.waist_yaw_joint_indices)
        print("waist_roll_joint_indices",self.waist_roll_joint_indices)

        self.left_arm_joint_indices = torch.zeros(len(self.cfg.asset.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.asset.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_arm_joints[i])
            
        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])
            
        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])
        self.torso_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.torso_link)
        self.pelvis_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.pelvis_link)
        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link)
        self.imu_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link)

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)

    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == 'plane':
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            return torch.cat((left_foot_height, right_foot_height), dim=-1), torch.cat((left_foot_height_var, right_foot_height_var), dim=-1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points/self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points/self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0]-2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0]-2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1]-2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1]-2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px+1, left_py]
        left_heights3 = self.height_samples[left_px, left_py+1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_foot_heights =  left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px+1, right_py]
        right_heights3 = self.height_samples[right_px, right_py+1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_foot_heights =  right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat((torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)), dim=-1)
        feet_heights_var = torch.cat((torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)), dim=-1)

        return torch.clip(feet_heights, min=0.), feet_heights_var

    def _compute_support_band_margin(self, com_xy, left_xy, right_xy, half_width):
        """
    com_xy:   (N,2) CoM 投影
    left_xy:  (N,2) 左脚中心投影
    right_xy: (N,2) 右脚中心投影
    half_width: 标量，支撑带半宽
    return:
        margin: (N,) >0 在支撑带内，<0 在外
     """
    # 中线方向（两脚连线）
        v = right_xy - left_xy                              # (N,2)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
        t = v / v_norm                                      # (N,2) unit tangent along foot-to-foot line

        # 法向量 n（旋转 90°）
        n = torch.stack([-t[:, 1], t[:, 0]], dim=-1)        # (N,2) unit normal

        # CoM 到中线的有符号法向距离：d = (com - mid) dot n
        mid = 0.5 * (left_xy + right_xy)
        d = torch.sum((com_xy - mid) * n, dim=-1).abs()     # (N,)

    # 支撑带裕度：带内为正，越靠近边界越小
        margin = half_width - d                             # (N,)
        return margin

    #------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )        
        h_target = h_ref+self.commands[:,4]
        return torch.square(self.base_lin_vel[:, 2]) *  (h_target >= 0.735)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions[:,:29]), dim=1)
    
    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        height_error = torch.abs(base_height - h_target+ self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error * 4)
    
    def _reward_stability_margin(self):
        """
        基于 CoM 投影相对支撑带的裕度 margin 的稳定性惩罚（返回负值更直观也行）
        建议在总 reward 里加：  + self.cfg.rewards.stability * r_stab
        或者直接返回“误差项”再乘负号。
    """
     # 1) CoM 近似：先用 base/root 位置投影（更严谨可用刚体质量加权 CoM）
        com_xy = self.root_states[:, 0:2]                   # (N,2)

        # 2) 脚中心（你已经有 feet_pos）
        # feet_pos: (N, num_feet, 3) 这里假设 num_feet=2
        left_xy  = self.feet_pos[:, 0, 0:2]
        right_xy = self.feet_pos[:, 1, 0:2]

    # 3) 只在“双脚都接触”时使用该指标（否则支撑带定义不可靠）
    # contact_forces: (N, num_bodies, 3) 你之前这样算过 contact
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0  # (N,2)
        double_support = (contact[:, 0] & contact[:, 1]).float()  # (N,)

    # 4) 计算支撑带裕度
        margin = self._compute_support_band_margin(
            com_xy, left_xy, right_xy, self.stab_band_half_width
        )  # (N,)

    # 5) 软惩罚：只有当 margin < m_safe 才惩罚
    # err = ReLU(m_safe - margin)
        err = torch.relu(self.stab_m_safe - margin) * double_support

        # 6) extend_dist gating：需要弯腰时惩罚更强
        # extend_dist 你是 (N,1) 或 (N,) 都可能，这里统一 squeeze
        ext = self.extend_dist.squeeze(-1) if self.extend_dist.dim() > 1 else self.extend_dist
        need_waist = (ext > 1e-3).float()  # (N,)

        # w = self.stab_w1 * need_waist       # (N,)

    # 返回“负惩罚”或“误差”都行，这里返回负惩罚（越大越好）
        r_stab = need_waist * err

    # 你可以顺便把诊断量写到 extras（方便 TB 看是否真的前冲）


        return r_stab    
    # def _reward_deviation_arm_joint(self):
    #     # arm_hold_weight = 1.0 - self.arm_curriculum.value(self.common_step_counter) 
    #     # print(self.default_dof_pos[:,self.arm_joint_indices])
    #     # print(self.arm_joint_indices)
    #     arm_dev = torch.sum(
    #     torch.square(
    #         self.dof_pos-self.default_dof_pos)[:,self.arm_joint_indices]
    #     ,
    #     dim=-1
    # )
    #     target_mask = self.is_target.float()
    #     # print(target_mask)
    #     arm_reward = (1.0 - target_mask) * arm_dev
    #     # print(torch.sum(arm_reward))
    #     return torch.sum(arm_reward,dim=-1)
    def _reward_reach_success(self):
        arm_mask = (self.is_target.squeeze(-1)).float()  # 或 target_arm!=0
        reachable_mask = (self.extend_dist.squeeze(-1) <= 0.03).float()
        d = torch.norm(self.target_position - self.ee_pos,dim=-1)
        # print("success rate:", (d < self.d_limit).float().mean().item())        
        return (d<self.d_limit).float()* arm_mask*reachable_mask
    
    def _reward_reach_target(self):
        arm_mask = (self.is_target.squeeze(-1)).float()  # 或 target_arm!=0
        d = torch.norm(self.target_position - self.ee_pos, dim=-1)
        return torch.exp(-10.0 * d)*arm_mask
    
    def _reward_reach_progress(self):

        arm_mask = (self.is_target.squeeze(-1)).float()
        d_now = torch.norm(self.target_position - self.ee_pos,dim=-1)
        d_prev = torch.norm(self.target_position - self.last_ee_pos,dim=-1)  # 你需要维护 last_dist2ee
        prog = (d_prev - d_now)
        # prog = torch.clamp(prog,)
        # clip 防止爆
        # prog = torch.clamp(prog,0, 0.05)
        return prog * arm_mask

    def _reward_deviation_roll_joint(self):
        """惩罚腰部在 roll 方向的弯曲"""
       # 获取腰部 roll 关节的角度
        waist_roll_angle = self.dof_pos[:, self.waist_roll_joint_indices]        
        return torch.sum(-torch.exp(torch.abs(waist_roll_angle) * 5.0) )
    
    def _reward_deviation_waist_joint(self):
        waist_mask = self.is_waist.float()
        reach_mask =self.is_reachable.float()
        need_waist = self.need_waist.float()
        # waist_pitch_angle = self.dof_pos[:, self.waist_pitch_joint_indices] 
        wasit_action_min = self.default_dof_pos[:, self.waist_pitch_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.waist_pitch_joint_indices]
        waist_action_max = self.default_dof_pos[:, self.waist_pitch_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.waist_pitch_joint_indices]
        joint_deviation = (self.dof_pos[:, self.waist_pitch_joint_indices] - wasit_action_min) / (waist_action_max - wasit_action_min) # always positive
        l = 0.2
        x = torch.clamp(self.extend_dist / l,0.0,torch.sin(torch.tensor(0.52,device=self.extend_dist.device)))

        speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        gate_stop = torch.exp(-(speed_xy/0.02).pow(2))
        target_waist = torch.clamp(torch.asin(x) / 0.52, 0.0, 1.0)*0.5+0.5
        # print(target_waist)
        waist_vel = self.dof_vel[:, self.waist_pitch_joint_indices].squeeze(-1).abs()
        gate_waist =0.6+0.4* torch.clamp(waist_vel / 0.01, 0.0, 1.0)   # 腰在动才给 dist 梯度
        # print(gate_waist)
        # print(gate_stop)
        loss_dist =gate_stop*gate_waist* (self.extend_dist/0.15).pow(2)
        w= torch.clamp(self.extend_dist/0.15,0.0,1.0)
        waist_loss =  torch.relu(target_waist-joint_deviation)*waist_mask*(1-reach_mask)*need_waist*130.5
        zero_loss = torch.abs(joint_deviation-0.5)*(1-need_waist)*waist_mask*0.55
        dist_loss = loss_dist*waist_mask*need_waist*2.2
        first_step_loss =  torch.abs(joint_deviation-0.5)*(1-waist_mask)*0.01
        # first_loss = -torch.exp(torch.abs(waist_roll_angle) * 5.0)

        out = torch.cat([waist_loss,zero_loss,dist_loss],dim=-1)
        # print(out)
        
        # print(loss_dist)
        return torch.sum(waist_loss+zero_loss+dist_loss+first_step_loss,dim=-1)

    
    def _reward_deviation_hip_joint(self):
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1) *  (h_target >= 0.735)
    
    def _reward_deviation_ankle_joint(self):
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref


        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) *  (h_target >= 0.735)
    
    def _reward_deviation_knee_joint(self):
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        height_error = self.root_states[:,2]-h_target
        # height_error = 0
        knee_action_min = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        knee_action_max = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (knee_action_max - knee_action_min) # always positive
        return torch.sum(torch.abs((joint_deviation-0.5) * height_error.unsqueeze(-1)), dim=-1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, :self.num_actions].clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, :self.num_actions].clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 # no reward for zero command
        return rew_airTime
    
    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var = self._get_feet_heights()
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        return torch.sum(height_error * feet_leteral_vel, dim=1) * ((h_target)>=0.71)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(-foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0) * ((h_target)>= 0.735)
    
    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1])
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(-knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0) * ((h_target) >= 0.735)
    
    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)
    
    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        return feet_distances_var * ((h_target) >= 0.735)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions[:,:29] - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + 0.2 * torch.square(self.commands[:, 2]), min=0.1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, :]), dim=1)#self.num_lower_dof

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :]), dim=1)#self.num_lower_dof
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit)[:, :].clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit)[:, :].clip(min=0.), dim=1)#self.num_lower_dof
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :3], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target[:, :] - self.dof_pos[:, :]), dim=-1)#self.num_lower_dof
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_action_vanish(self):
        upper_error = torch.clip(self.origin_actions[:, :] - self.action_max[:, :], min=0)
        lower_error = torch.clip(self.action_min[:, :] - self.origin_actions[:, :], min=0)
        return torch.sum(upper_error + lower_error, dim=-1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)
        h_ref = torch.where(
        self.target_position[:,2] < 0.0,
        torch.clamp(self.target_position[:,2]+0.64,0.25,0.74),
        0.74
        )     
        mask = self.is_target_height.view(-1)   # [N]，不要 [N,1]
   
        h_target = (1-mask)*self.commands[:,4]+mask*h_ref
        error_sim = (contacts) * ((h_target) >= 0.735)
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)