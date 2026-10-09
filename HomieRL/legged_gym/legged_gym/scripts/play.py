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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import onnxruntime as ort

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import numpy as np
import torch
import time

def load_policy():
    body = torch.jit.load("", map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        print(action.shape)
        return action
    return policy

def load_onnx_policy():
    model = ort.InferenceSession("")
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False
    env_cfg.env.is_trainning =False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    # env.is_target[:]=0
    # env.is_reachable[:]=0
    # env.target_arm[:]=0


    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    # policy = load_onnx_policy() # Use this to load from exported onnx file
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    num_steps = 0
    for _ in range(10*int(env.max_episode_length)):
        num_steps+=1
        # print(num_steps)
        env.action_curriculum_ratio = 1.0
        actions = policy(obs.detach())
        # if num_steps>50 and num_steps<300:
        #     env.target_position[:, 0] = 0.1+(num_steps/250)*0.6
        #     env.target_position[:, 1] = 0.1
        #     env.target_position[:,2] = 0.1
        #     env.is_target_height[:] = 1
        #     env.target_arm[:, 0] = 1
        #     env.is_target[:] = 0
        #     env.is_waist[:] = 1
        # if num_steps>350 and num_steps<600:
        #     env.target_position[:, 0] = 0.1+((num_steps-350)/250)*0.6
        #     env.target_position[:, 1] = 0.1
        #     env.target_position[:,2] = -0.2
        #     env.is_target_height[:] = 1
        #     env.target_arm[:, 0] = 1
        #     env.is_target[:] = 0
        #     env.is_waist[:] = 1
        # if num_steps>650 and num_steps<900:
        # env.target_position[:, 0] = 0.1
        # env.target_position[:, 1] = 0.1
        # env.target_position[:,2] = 0.2
        if num_steps>50 and num_steps<300:

            env.is_target_height[:] = 1
            env.target_arm[:, 0] = 1
            env.is_target[:] = 1
            env.is_waist[:] = 1
            env.target_position[:, 0] = (0.5/250)*(num_steps-50)
            env.target_position[:, 1] = (0.1/250)*(num_steps-50)
            env.target_position[:, 2] = (-0.2/250)*(num_steps-50)
        # roll = torch.tensor(60, device=env.device)
        # pitch = torch.tensor(0.0, device=env.device)
        # yaw = torch.tensor(-90, device=env.device)
        # env.target_rot[:]=quat_from_euler_xyz(roll,pitch,yaw)
        # env.target_rot[:,0] = 1.57
        # env.target_rot[:,1] = -1.57
        # env.target_rot[:,2] = 0
    #     if num_steps<150:
    #         env.commands[:, 0] = 1.0
    #         env.commands[:, 1] = y_vel
    #         env.commands[:, 2] = yaw_vel
    #         env.commands[:, 4] = height
    #         env.is_target_height[:] = 0
    #         env.target_arm[:, 0] = 0
    #         env.is_target[:] = 0
    #         env.is_waist[:] = 0
    #         # env.target_position[:, 0] = 0.0
    #         # env.target_position[:, 1] = 0.0
    #         # env.target_position[:,2] = 0.0
    #     if num_steps>150 and num_steps<170:
    #         env.commands[:, 0] = 0.0
    #         env.commands[:, 1] = y_vel
    #         env.commands[:, 2] = yaw_vel
    #         env.commands[:, 4] = height
    #     elif num_steps>170 and num_steps<=520:
    #         env.target_position[:, 0] = 0.1+((num_steps-170)/300)*0.55
    #         env.target_position[:, 1] = 0.15
    #         # env.target_position[:,2] = -0.5
    #         env.is_target_height[:] = 1
    #         env.target_arm[:, 0] = 1
    #         env.is_target[:] = 0
    #         env.is_waist[:] = 1

    # # Initialize target position to zero
    #         # env.target_position[:, 0] = 0.2+(num_steps/500)*0.55
    #         # env.target_position[:, 1] = 0.1
    #         env.target_position[:,2] = 0.24 - ((num_steps-170)/350)*0.64
    # #     elif num_steps>550 and num_steps<=1100:
    # # # ===== 1. 设置目标高度 =====
    #         env.target_position[:, 0] = 0.2+((num_steps-500)/500)*0.55
    #         env.target_position[:, 1] = 0.1
    #         env.target_position[:,2] = -0.2
    #     elif num_steps>1150 :
    #         env.target_position[:, 0] = 0.2+((num_steps-1100)/500)*0.55
    #         env.target_position[:, 1] = 0.1
        # env.target_position[:,2] = -0.4

        # env.target_position[:, 2] = 0.2-(num_steps/500)*0.9
        # torch.clamp(env.target_position[:, 2],-0.7,0.2)
        # print(env.target_position[:, 2])
    # Set flags for target height and arm

    
        # print("cmd:", env.commands[0, :5].cpu().numpy(), "is_waist:", env.is_waist[0].item(), "is_target:", env.is_target[0].item())

        obs, _, _, _, _, _, _ = env.step(actions.detach())
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74)