#!/home/eisr/miniconda3/envs/goal/bin/python3
import sys
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
from legged_gym import LEGGED_GYM_ROOT_DIR
import teleop_pkg
from teleop_pkg.robot_control import robot_arm_ik
import threading
import  time 
import math
import csv
import os
last_arm_target_positions = np.zeros(14,dtype=np.float32)
pending_target = None
csv_path = "./wcgs.csv"

class WCGSEvaluator:
    """
    Whole-body Coordinated Grasping Score evaluator for MuJoCo experiments.

    记录单个目标点实验过程中的状态，并在 episode 结束后计算:
    S_grasp, S_reach, S_stab, S_smooth, S_body, WCGS
    """

    def __init__(
        self,
        alpha=5.0,
        gamma=5.0,
        eta=10.0,
        beta=5.0,
        arm_length=0.50,
        dex_max=0.20,
        waist_k=3.0,
        waist_max=0.5,
        waist_forward_sign=1.0,
        grasp_success_threshold=0.3,
        weights=None,
    ):
        """
        Args:
            alpha: 抓取误差惩罚系数
            gamma: 稳定性惩罚系数
            eta: 动作平滑性惩罚系数
            beta: 腰部补偿误差惩罚系数

            arm_length: 手臂最大可达长度, 例如 0.50 m
            dex_max: 最大超出可达域距离, 用于归一化, 例如 0.20 m

            waist_k: d_ex 到参考腰部角的比例系数
            waist_max: 腰部最大参考补偿角, 单位 rad
            waist_forward_sign:
                腰部前倾方向符号。
                如果 MuJoCo 里 waist_pitch 正值表示前倾, 设为 1.0；
                如果负值表示前倾, 设为 -1.0。

            grasp_success_threshold: 判断抓取成功的末端误差阈值

            weights: WCGS 各项权重
        """

        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.beta = beta

        self.arm_length = arm_length
        self.dex_max = dex_max
        self.waist_k = waist_k
        self.waist_max = waist_max
        self.waist_forward_sign = waist_forward_sign
        self.grasp_success_threshold = grasp_success_threshold

        if weights is None:
            self.weights = {
                "grasp": 0.40,
                # "reach": 0.20,
                "stab": 0.40,
                "smooth": 0.10,
                "body": 0.10,
            }
        else:
            self.weights = weights

        self.reset()

    def reset(self):
        self.ee_pos_list = []
        self.target_pos_list = []
        self.shoulder_pos_list = []
        self.pelvis_pos_list = []
        self.midfeet_pos_list = []
        self.action_list = []
        self.waist_pitch_list = []

    def record(
        self,
        ee_pos,
        target_pos,
        shoulder_pos,
        pelvis_pos,
        left_foot_pos,
        right_foot_pos,
        action,
        waist_pitch,
    ):
        """
        每个仿真 step 调用一次。

        Args:
            ee_pos: 当前末端位置, shape=(3,)
            target_pos: 当前目标位置, shape=(3,)
            shoulder_pos: 当前肩部位置, shape=(3,)
            pelvis_pos: 当前骨盆位置, shape=(3,)
            left_foot_pos: 左脚位置, shape=(3,)
            right_foot_pos: 右脚位置, shape=(3,)
            action: 当前控制输入, shape=(action_dim,)
            waist_pitch: 当前腰部 pitch 角
        """

        ee_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        shoulder_pos = np.asarray(shoulder_pos, dtype=np.float64).copy()
        pelvis_pos = np.asarray(pelvis_pos, dtype=np.float64).copy()
        left_foot_pos = np.asarray(left_foot_pos, dtype=np.float64).copy()
        right_foot_pos = np.asarray(right_foot_pos, dtype=np.float64).copy()
        action = np.asarray(action, dtype=np.float64).copy()

        midfeet_pos = 0.5 * (left_foot_pos + right_foot_pos)

        self.ee_pos_list.append(ee_pos)
        self.target_pos_list.append(target_pos)
        self.shoulder_pos_list.append(shoulder_pos)
        self.pelvis_pos_list.append(pelvis_pos)
        self.midfeet_pos_list.append(midfeet_pos)
        self.action_list.append(action)
        self.waist_pitch_list.append(float(waist_pitch))

    def compute(self):
        """
        episode 结束后调用，返回一个 dict。
        """

        if len(self.ee_pos_list) == 0:
            raise RuntimeError("WCGSEvaluator has no recorded data.")

        ee_pos = np.asarray(self.ee_pos_list)
        target_pos = np.asarray(self.target_pos_list)
        shoulder_pos = np.asarray(self.shoulder_pos_list)
        pelvis_pos = np.asarray(self.pelvis_pos_list)
        midfeet_pos = np.asarray(self.midfeet_pos_list)
        actions = np.asarray(self.action_list)
        waist_pitch = np.asarray(self.waist_pitch_list)

        # =========================================================
        # 1. 抓取精度 S_grasp
        # =========================================================
        ee_error_each_step = np.linalg.norm(ee_pos - target_pos, axis=1)-0.14

        # 建议用最后一段稳定阶段的平均误差，而不是只用最后一帧
        tail_len = max(1, len(ee_error_each_step) // 5)
        ee_error_final = float(np.mean(ee_error_each_step[-tail_len:]))

        S_grasp = float(np.exp(-self.alpha * ee_error_final))

        success = float(ee_error_final < self.grasp_success_threshold)

        # =========================================================
        # 2. 跨可达域能力 S_reach
        # =========================================================
        target_shoulder_dist_each_step = np.linalg.norm(target_pos - shoulder_pos, axis=1)

        # # 同样取最后稳定阶段
        target_shoulder_dist = float(np.mean(target_shoulder_dist_each_step[-tail_len:]))

        d_ex = max(0.0, target_shoulder_dist - self.arm_length)
        d_ex_norm = np.clip(d_ex / self.dex_max, 0.0, 1.0)

        # # 只有抓得好，跨可达域才给高分
        # S_reach = float(d_ex_norm * S_grasp)

        # =========================================================
        # 3. 稳定性 S_stab
        # =========================================================
        pelvis_xy = pelvis_pos[:, :2]
        midfeet_xy = midfeet_pos[:, :2]

        stab_dist_each_step = np.linalg.norm(pelvis_xy - midfeet_xy, axis=1)

        # 可以取全程平均，也可以取最后稳定阶段平均
        stab_dist_mean = float(np.mean(stab_dist_each_step))
        stab_dist_final = float(np.mean(stab_dist_each_step[-tail_len:]))

        S_stab = float(np.exp(-self.gamma * stab_dist_mean))

        # =========================================================
        # 4. 动作平滑性 S_smooth
        # =========================================================
        if len(actions) >= 2:
            action_diff = actions[1:] - actions[:-1]
            J_smooth = float(np.mean(np.sum(action_diff ** 2)))
        else:
            J_smooth = 0.0

        S_smooth = float(np.exp(- J_smooth))

        # =========================================================
        # 5. 身体补偿合理性 S_body
        # =========================================================
        # 根据几何可达性计算参考腰部角
        waist_ref_abs = np.clip(self.waist_k * d_ex, 0.0, self.waist_max)
        waist_ref = self.waist_forward_sign * waist_ref_abs

        waist_pitch_final = float(np.mean(waist_pitch[-tail_len:]))
        waist_error = abs(waist_pitch_final - waist_ref)

        S_body = float(np.exp(-self.beta * waist_error))

        # =========================================================
        # 6. WCGS 总分
        # =========================================================
        WCGS = (
            self.weights["grasp"] * S_grasp
            # + self.weights["reach"] * S_reach
            + self.weights["stab"] * S_stab
            + self.weights["smooth"] * S_smooth
            + self.weights["body"] * S_body
        )

        result = {
            "WCGS": float(WCGS),

            "S_grasp": S_grasp,
            # "S_reach": S_reach,
            "S_stab": S_stab,
            "S_smooth": S_smooth,
            "S_body": S_body,

            "ee_error_final": ee_error_final,
            "success": success,

            "target_shoulder_dist": target_shoulder_dist,
            "d_ex": float(d_ex),
            "d_ex_norm": float(d_ex_norm),

            "stab_dist_mean": stab_dist_mean,
            "stab_dist_final": stab_dist_final,

            "J_smooth": J_smooth,

            "waist_pitch_final": waist_pitch_final,
            "waist_ref": float(waist_ref),
            "waist_error": float(waist_error),

            "num_steps": len(self.ee_pos_list),
        }

        return result

def save_wcgs_result_to_csv(csv_path, result_dict):
    """
    保存单个目标点的 WCGS 结果。
    如果文件不存在，则自动写表头。
    """

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_dict.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(result_dict)

shared_state = {
    "lock": threading.Lock(),
    "step": 0,
    "q_arm": None,
    "target": None,
    "ik_step": -1,
    "arm_target": None,
    "running": True,
}


def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    # Convert lists to numpy arrays where needed
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    
    return config

def set_table_and_ball_pose(m, d, ball_pos):
    """
    ball_pos: 目标球世界坐标 [x, y, z]
    """

    ball_pos = np.asarray(ball_pos, dtype=np.float64)

    # =========================
    # 1. 设置桌子位置
    # =========================
    table_body_id = m.body("target_table").id

    table_pos = ball_pos.copy()
    table_pos[2] = ball_pos[2]  - 0.03
    # table_pos[0]-=0.05
    m.body_pos[table_body_id] = table_pos

    # =========================
    # 2. 设置球体位置
    # =========================
    ball_jid = m.joint("target_bottle_freejoint").id
    ball_qpos_adr = m.jnt_qposadr[ball_jid]
    ball_qvel_adr = m.jnt_dofadr[ball_jid]

    d.qpos[ball_qpos_adr:ball_qpos_adr + 3] = ball_pos
    d.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    d.qvel[ball_qvel_adr:ball_qvel_adr + 6] = 0.0

    mujoco.mj_forward(m, d)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    # print(kp.shape,kd.shape)
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def get_extend_dist(target_pos,d):
    if target_pos[1] >= 0:
        right_shoulder_pos = d.xpos[d.body('right_shoulder_pitch_link').id]
        vec = target_pos - right_shoulder_pos
        dist = np.linalg.norm(vec)
    elif target_pos[1] < 0:
        left_shoulder_pos = d.xpos[d.body('left_shoulder_pitch_link').id]
        vec = target_pos - left_shoulder_pos
        dist = np.linalg.norm(vec)
    arm_length = 0.46
    extend_dist = max(0.0,dist-arm_length)
    return extend_dist

def ik_dls_single(J_pos, pos_err, q_arm, damping=0.01, alpha=1.0):
    """
    J_pos: np.array [3,n]
    pos_err: np.array [3,]
    q_arm: np.array [n,]
    """
    J_pos_np = J_pos.numpy() if isinstance(J_pos, torch.Tensor) else J_pos
    pos_err_np = pos_err.numpy() if isinstance(pos_err, torch.Tensor) else pos_err
    q_arm_np = q_arm.numpy() if isinstance(q_arm, torch.Tensor) else q_arm

    JJt = J_pos_np @ J_pos_np.T
    I3 = np.eye(3)
    A = JJt + (damping ** 2) * I3
    x = np.linalg.solve(A, pos_err_np)
    qdot = J_pos_np.T @ x
    q_des = q_arm_np + alpha * qdot
    return q_des


def ik_from_q_realtime(target_pos, arm_target_positions, d):
    """
    实时渐进式 IK：
    每步根据当前末端误差计算 dq，
    但不是 q_des = q_current + dq，
    而是 q_ref = q_ref + dq。

    target_pos: [3,] 世界坐标系目标
    arm_target_positions: [14,] 上一帧手臂目标角，左7 + 右7
    d: mujoco.MjData

    return:
        new_arm_target_positions: [14,]
        ee_error: float
    """
    import numpy as np
    import mujoco

    model = d.model
    target_pos = np.asarray(target_pos, dtype=np.float64)

    # 和 get_extend_dist 保持一致：
    # y >= 0 -> right arm
    # y < 0  -> left arm
    if target_pos[1] <0:
        arm_offset = 7
        ee_body_name = "right_wrist_yaw_link"
        joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
    else:
        arm_offset = 0
        ee_body_name = "left_wrist_yaw_link"
        joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]

    ee_body_id = model.body(ee_body_name).id

    joint_ids = [model.joint(name).id for name in joint_names]
    qpos_ids = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    dof_ids = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)

    # 当前真实关节角，只用于计算当前状态
    q_current = d.qpos[qpos_ids].copy()

    # 上一帧目标角，作为连续参考
    q_ref = arm_target_positions[arm_offset:arm_offset+7].copy()

    # 当前真实末端误差
    ee_pos = d.xpos[ee_body_id].copy()
    pos_err = target_pos - ee_pos
    ee_error = np.linalg.norm(pos_err)

    # 目标附近设置死区，避免来回抖动
    if ee_error < 0.02:
        return arm_target_positions.copy(), ee_error

    # 任务空间误差限幅：远目标每步只追一小段
    max_pos_step = 0.015
    if ee_error > max_pos_step:
        pos_err = pos_err / ee_error * max_pos_step

    # 当前真实姿态下的 Jacobian
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, d, jacp, jacr, ee_body_id)

    J = jacp[:, dof_ids].copy()

    if np.linalg.norm(J) < 1e-8:
        print("Warning: Jacobian near zero.")
        return arm_target_positions.copy(), ee_error

    # DLS
    damping = 0.08
    JJt = J @ J.T
    A = JJt + damping ** 2 * np.eye(3)

    try:
        x = np.linalg.solve(A, pos_err)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(A) @ pos_err

    dq = J.T @ x

    # 关键：每步关节目标变化要很小
    dq_limit_vec = np.ones(7, dtype=np.float64) * 0.008

# shoulder_roll 是单臂内部第 1 个索引
# 对它更严格，减少左右摆动
    dq_limit_vec[1] = 0.0015

    dq = np.clip(dq, -dq_limit_vec, dq_limit_vec)

# 用上一帧参考角更新
    q_new_ref = q_ref + dq

# 限制 q_ref 不要离真实关节太远
    max_ref_error_vec = np.ones(7, dtype=np.float64) * 0.08

# shoulder_roll 更严格，避免 PD 追不上导致晃动
    max_ref_error_vec[1] = 0.025

    q_new_ref = np.clip(
    q_new_ref,
    q_current - max_ref_error_vec,
    q_current + max_ref_error_vec
)

    # 关节限位
    for i, jid in enumerate(joint_ids):
        if model.jnt_limited[jid]:
            low, high = model.jnt_range[jid]
            q_new_ref[i] = np.clip(q_new_ref[i], low, high)

    new_arm_target_positions = arm_target_positions.copy()
    new_arm_target_positions[arm_offset:arm_offset+7] = q_new_ref

    return new_arm_target_positions, ee_error


def check_body_stable_by_waist_height(
    d,
    prev_waist_pitch,
    prev_height,
    stable_counter,
    waist_delta_threshold=0.003,
    height_delta_threshold=0.003,
    stable_required_steps=30,
):
    """
    判断身体姿态是否稳定：
    不是和给定目标比较，而是看腰部 pitch 和身体高度自身变化是否足够小。
    连续 stable_required_steps 次变化都小于阈值，认为稳定。
    """

    waist_pitch_idx = 14

    current_height = d.qpos[2]
    current_waist_pitch = d.qpos[7 + waist_pitch_idx]

    waist_delta = abs(current_waist_pitch - prev_waist_pitch)
    height_delta = abs(current_height - prev_height)

    frame_stable = (
        waist_delta < waist_delta_threshold
        and height_delta < height_delta_threshold
    )

    if frame_stable:
        stable_counter += 1
    else:
        stable_counter = 0

    body_stable = stable_counter >= stable_required_steps

    return (
        body_stable,
        stable_counter,
        current_waist_pitch,
        current_height,
        waist_delta,
        height_delta,
    )

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

def get_finger_actuator_ids(m):
    left_finger_joint_names = [
        "left_thumb_1_joint",
        "left_thumb_2_joint",
        "left_thumb_3_joint",
        "left_thumb_4_joint",
        "left_index_1_joint",
        "left_index_2_joint",
        "left_middle_1_joint",
        "left_middle_2_joint",
        "left_ring_1_joint",
        "left_ring_2_joint",
        "left_little_1_joint",
        "left_little_2_joint",
    ]

    right_finger_joint_names = [
        "right_thumb_1_joint",
        "right_thumb_2_joint",
        "right_thumb_3_joint",
        "right_thumb_4_joint",
        "right_index_1_joint",
        "right_index_2_joint",
        "right_middle_1_joint",
        "right_middle_2_joint",
        "right_ring_1_joint",
        "right_ring_2_joint",
        "right_little_1_joint",
        "right_little_2_joint",
    ]

    def find_act_ids(joint_names):
        act_ids = []
        for jname in joint_names:
            jid = m.joint(jname).id
            found = False

            for aid in range(m.nu):
                if m.actuator_trnid[aid, 0] == jid:
                    act_ids.append(aid)
                    found = True
                    break

            if not found:
                print(f"[WARN] no actuator found for {jname}")

        return np.array(act_ids, dtype=np.int32)

    left_finger_act_ids = find_act_ids(left_finger_joint_names)
    right_finger_act_ids = find_act_ids(right_finger_joint_names)

    return left_finger_act_ids, right_finger_act_ids

def get_joint_indices(m, joint_names):
    joint_ids = [m.joint(name).id for name in joint_names]
    qpos_ids = np.array([m.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    dof_ids = np.array([m.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
    return joint_ids, qpos_ids, dof_ids

def get_actuator_ids_for_joints(m, joint_names):
    actuator_ids = []

    for jname in joint_names:
        jid = m.joint(jname).id
        found = False

        for aid in range(m.nu):
            # actuator_trnid[aid, 0] 通常是该 actuator 绑定的 joint id
            if m.actuator_trnid[aid, 0] == jid:
                actuator_ids.append(aid)
                found = True
                break

        if not found:
            raise RuntimeError(f"No actuator found for joint: {jname}")

    return np.array(actuator_ids, dtype=np.int32)

def add_target_marker(viewer, pos):
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.04, 0.04, 0.04]),
        pos=np.array(pos),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.0, 0.0, 1.0])
    )
    viewer.user_scn.ngeom += 1

def compute_observation(d, config, action, cmd, height_cmd, n_joints,extend_dist,counter):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    qj = d.qpos[7:7+n_joints].copy()
    dqj = d.qvel[6:6+n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    
    # Handle default angles padding
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]
    
    # Scale the values
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    is_target = config['is_target']
    is_waist = config['is_waist']
    is_target_height = config['is_target_height']
    target_position = config['target_position']
    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 23 +16
    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation          
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+23] = action
    single_obs[10+2*n_joints+23:10+2*n_joints+23+1] = is_target
    single_obs[10+2*n_joints+24:10+2*n_joints+24+1] = is_waist
    single_obs[10+2*n_joints+25:10+2*n_joints+25+1] = extend_dist#extend_dist
    single_obs[10+2*n_joints+26:10+2*n_joints+26+1] = is_target_height
    progress = np.clip(counter / 3000.0, 0.0, 1.0)
    single_obs[10+2*n_joints+27:10+2*n_joints+27+1] = (target_position[2]+0.13)*progress
    single_obs[10+2*n_joints+28:10+2*n_joints+28+1] = 0 #target_arm
    single_obs[10+2*n_joints+29:10+2*n_joints+29+3] = [0.0,0.0,0.0] #target_arm
    single_obs[10+2*n_joints+32:10+2*n_joints+32+3] = [0.0,0.0,0.0] #target_arm
    single_obs[10+2*n_joints+35:10+2*n_joints+35+4] = [0.0,0.0,0.0,0.0] #target_arm

    
    return single_obs, single_obs_dim

def main():
    # Load configuration
    config = load_config("g1.yaml")
    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    n_joints = d.qpos.shape[0] - 7-7-24
    left_finger_act_ids, right_finger_act_ids = get_finger_actuator_ids(m)
    print("left_finger_act_ids =", left_finger_act_ids)
    print("right_finger_act_ids =", right_finger_act_ids)
    # Check number of joints
    # print(f"Robot has {n_joints} joints in MuJoCo model")
    target_position_read = config['target_position'].copy()
    target_position = np.array([0.0,0.0,0.75])
    target_pos = target_position_read.copy()
    target_pos[2]+=0.75
    set_table_and_ball_pose(m, d, target_pos)
    target_arm = target_position_read.copy()
    ee_pos_first = target_arm.copy()
    ee_pos_first[0] = 0.15
    ee_pos_first[2] = 0.83
    if target_position_read[1]>0:
        ee_pos_first[1] = 0.15
    else :
        ee_pos_first[1] = -0.15
    update_ee_pos = False
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    height_cmd = config['height_cmd']
    extend_dist = get_extend_dist(target_position,d)

    left_joint_ids, left_arm_qpos_ids, left_arm_dof_ids = get_joint_indices(m, LEFT_ARM_JOINTS)
    right_joint_ids, right_arm_qpos_ids, right_arm_dof_ids = get_joint_indices(m, RIGHT_ARM_JOINTS)

    left_arm_act_ids = get_actuator_ids_for_joints(m, LEFT_ARM_JOINTS)
    right_arm_act_ids = get_actuator_ids_for_joints(m, RIGHT_ARM_JOINTS)

    # d.ctrl[22] = -1.3
    # d.ctrl[23] = 0.0
    # d.ctrl[24] = 0.5
    # d.ctrl[25] = 0.0
    # d.ctrl[26] = 0.0
    # d.ctrl[27] = 0.0
    # d.ctrl[28] = 0.0
    # d.ctrl[29] = 0.0
    # d.ctrl[30] = 0.0
    # d.ctrl[31] = 0.0
    # d.ctrl[32] = 0.0
    # d.ctrl[33] = 0.0
    # d.ctrl[22] = 0.0
    # d.ctrl[23] = 0.8
    # d.ctrl[24] = 0.8
    # d.ctrl[26] = 1.0
    # # d.ctrl[27] = -0.5
    # d.ctrl[28] = 0.8
    # # d.ctrl[29] = -0.5
    # d.ctrl[30] = 0.9
    # # d.ctrl[31] = 0.5
    # d.ctrl[32] = 0.9
    # d.ctrl[33] = -0.8



    arm_target_positions = None
    ee_error=0.1
    
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints,extend_dist,1)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    # get_extend_dist(target_position,d)

    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load policy
    policy = torch.jit.load(config['policy_path'])
    wcgs_evaluator = WCGSEvaluator(
    alpha=10.0,
    gamma=5.0,
    eta=10.0,
    beta=5.0,
    arm_length=0.50,
    dex_max=0.20,
    waist_k=3.0,
    waist_max=0.5,
    waist_forward_sign=1.0,   # 这里要根据你的 waist_pitch 前倾方向改
    grasp_success_threshold=0.3,
)
    counter = 0
    step = 0
    bend  =0.5
    left_arm_q = d.qpos[left_arm_qpos_ids].copy()
    right_arm_q =d.qpos[right_arm_qpos_ids].copy()
    arm_target_positions = np.concatenate([left_arm_q, right_arm_q]).astype(np.float32)
    new_arm_target_positions =arm_target_positions.copy()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            viewer.user_scn.ngeom = 0
            viewer.sync()
            step_start = time.time()
            ee_pos = d.xpos[d.body("right_wrist_yaw_link" if target_arm[1] < 0 else "left_wrist_yaw_link").id].copy()
            # print("target_arm",target_arm)
            ee_error = np.linalg.norm(target_pos - ee_pos)
            if ee_error>0.05 and counter%20==0 and step <=3000:
                        new_arm_target_positions, _ = ik_from_q_realtime(
    target_arm,
    arm_target_positions,
    d
)
            arm_target_positions = (1.0-bend)  *arm_target_positions + bend *new_arm_target_positions
            if counter % 50 == 0:
                print("Realtime IK ee_error:", ee_error)
                print("arm_target_positions:", arm_target_positions)      
                print("real_target_positions", d.qpos[7+15:7+n_joints])
                print("ee_pos",ee_pos)
                print("target_dof_pos",target_dof_pos[:12])
                print(target_arm)
            # print("Body stable, solve IK once.")
            # print("waist_delta:", prev_waist_pitch, "height_delta:", prev_height)
            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']-8],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']-8],
                config['kds']
            )
            
            d.ctrl[:12] = leg_tau[:12]
            waist_tau = pd_control(
                target_dof_pos[13:15],
                d.qpos[7+13:7+13+2],
                config['kps'][13:15],
                np.zeros_like(config['kps'])[13:15],
                d.qvel[6+13:6+13+2],
                config['kds'][13:15]
            )
            d.ctrl[13:15] = waist_tau
            
            # Keep other joints at zero positions if they exist
            if n_joints > config['num_actions']:
                arm_kp = 100.0
                arm_kd = 0.5
                
                left_arm_tau = pd_control(
                arm_target_positions[:7],
                d.qpos[left_arm_qpos_ids],
                np.ones(7) * arm_kp,
                np.zeros(7),
                d.qvel[left_arm_dof_ids],
                np.ones(7) * arm_kd,
                )
                right_arm_tau = pd_control(
                arm_target_positions[7:14],
                d.qpos[right_arm_qpos_ids],
                np.ones(7) * arm_kp,
                np.zeros(7),
                d.qvel[right_arm_dof_ids],
                np.ones(7) * arm_kd,
                )

                arm_tau1 = pd_control(
                    0,
                    d.qpos[7+12],
                    np.ones(1) * 300,
                    np.zeros(1),
                    d.qvel[6+12],
                    np.ones(1) * 5
                )
                d.ctrl[12] = arm_tau1
                # arm_tau2 = pd_control(
                #     0,
                #     d.qpos[7+14],
                #     np.ones(1) * 300,
                #     np.zeros(1),
                #     d.qvel[6+14],
                #     np.ones(1) * 5
                # )
                # d.ctrl[14] = arm_tau2                
                if d.ctrl.shape[0] > config['num_actions']:
                    d.ctrl[left_arm_act_ids] = left_arm_tau
                    d.ctrl[right_arm_act_ids] = right_arm_tau
            # print("angle",d.qpos[7+14])
            
            # Step physics
            mujoco.mj_step(m, d)

            counter += 1
            step += 1
            if counter % config['control_decimation'] == 0:
                # Update observation
                pelvis_pos = d.xpos[d.body('pelvis').id]
                left_wrist_pos = d.xpos[d.body('left_wrist_yaw_link').id]
                # print(left_wrist_pos)
                # print(extend_dist)
                if counter /2000==0:
                    update_ee_pos = True
                if update_ee_pos:
                    ee_pos_first = d.xpos[d.body("right_wrist_yaw_link" if target_arm[1] < 0 else "left_wrist_yaw_link").id].copy()
                    print("ee_pos_first",ee_pos_first)
                if step<2000:
                    target_position[0]=0.0+step/2000*(target_position_read[0]-0.0)
                    target_position[2]=step/2000*(target_position_read[2]-0.0)+0.75
                # print(target_position)
                extend_dist = get_extend_dist(target_position,d)
                if step<2000:
                    # d.ctrl[22] = 0.0
                    # d.ctrl[23] = 0.0
                    # d.ctrl[24] = 0.0
                    # d.ctrl[25] = 0.0
                    # d.ctrl[26] = 0.0
                    # d.ctrl[27] = 0.0
                    # d.ctrl[28] = 0.0
                    # d.ctrl[29] = 0.0
                    # d.ctrl[30] = 0.0
                    # d.ctrl[31] = 0.0
                    # d.ctrl[32] = 0.0
                    # d.ctrl[33] = 0.0
                    if target_position_read[0]>0.4:
                        b=np.clip(step/2000,0.0,1.0)
                        target_arm[0]=ee_pos_first[0]+(target_position_read[0]-ee_pos_first[0]-0.2)*b
                        # if target_position_read[1]<0:
                        #     target_arm[1]=-0.15+(target_position_read[1]+0.15)/2000*counter
                        # else :
                        alpha = np.clip(step/200,0.0,1.0)
                        target_arm[1]=ee_pos_first[1]+(target_position_read[1]-ee_pos_first[1]+0.12)*alpha

                        # a = math.fabs(target_position_read[2])
                    # a = a/0.15*0.2
                        b=np.clip(step/2500,0.0,1.0)
                        target_arm[2]=ee_pos_first[2]+(target_position_read[2]+0.15)*b
                        # elif step>=1500 and step<=2000:
                        #  b=np.clip(step/2000,0.0,1.0)
                        #  target_arm[2]=ee_pos_first[2]+(target_position_read[2]+0.05)*b
                    else :
                        b=np.clip(step/2000,0.0,1.0)
                        target_arm[0]=ee_pos_first[0]+(target_position_read[0]-ee_pos_first[0]+0.07)*b
                        # if target_position_read[1]<0:
                        #     target_arm[1]=-0.15+(target_position_read[1]+0.15)/2000*counter
                        alpha = np.clip(step/200,0.0,1.0)
                        target_arm[1]=ee_pos_first[1]+(target_position_read[1]-ee_pos_first[1]+0.07)*alpha
                        a = math.fabs(target_position_read[2])
                    # a = a/0.15*0.2
                        target_arm[2]=ee_pos_first[2]+(target_position_read[2]+0.1)/2000*step
                if step==2000:
                       ee_pos_first =  d.xpos[d.body("right_wrist_yaw_link" if target_arm[1] < 0 else "left_wrist_yaw_link").id].copy()
                if step >2000 and step <3000:
                    if target_position_read[0]>0.35:
                        
                        b=np.clip((step-2000)/1000,0.0,1.0)
                        target_arm[0]=ee_pos_first[0]+(target_position_read[0]-ee_pos_first[0]-0.065)*b
                        # if target_position_read[1]<0:
                        #     target_arm[1]=-0.15+(target_position_read[1]+0.15)/2000*counter
                        # else :
                        alpha = np.clip((step-2000)/1000,0.0,1.0)
                        target_arm[1]=ee_pos_first[1]+(target_position_read[1]-ee_pos_first[1]+0.045)*alpha

                        b=np.clip((step-2000)/1000,0.0,1.0)
                        target_arm[2]=ee_pos_first[2]+(target_position_read[2]+0.03)*b
                    else :
                        target_arm[0]=ee_pos_first[0]+(target_position_read[0]-ee_pos_first[0]-0.07)/2000*step
                        # if target_position_read[1]<0:
                        #     target_arm[1]=-0.15+(target_position_read[1]+0.15)/2000*counter
                        # else :
                        target_arm[1]=ee_pos_first[1]+(target_position_read[1]-ee_pos_first[1]+0.04)/2000*step
                        a = math.fabs(target_position_read[2])
                    # a = a/0.15*0.2
                        target_arm[2]=ee_pos_first[2]+(target_position_read[2]+0.05)/2000*step

                if step>3000:


                    d.ctrl[22] = 0.0
                    d.ctrl[23] = 1.4
                    d.ctrl[24] = 0.8
                    d.ctrl[26] = 1.4
                    # d.ctrl[27] = -0.5
                    d.ctrl[28] = 1.4
                    # d.ctrl[29] = -0.5
                    d.ctrl[30] = 1.4
                    # d.ctrl[31] = 0.5
                    d.ctrl[32] = 1.4
                    d.ctrl[33] = -0.9
                #     d.ctrl[22] = 0.0
                #     d.ctrl[23] = 1.4
                #     d.ctrl[24] = 0.8
                #     d.ctrl[26] = 1.4
                #     # d.ctrl[27] = -0.5
                #     d.ctrl[28] = 1.4
                #     # d.ctrl[29] = -0.5
                #     d.ctrl[30] = 1.4
                #     # d.ctrl[31] = 0.5
                #     d.ctrl[32] = 1.4
                #     d.ctrl[33] = -0.9
                # #     print(counter)
                #     step = 0
                #     target_position_read[0]=0.15
                #     target_position_read[1]=0.15
                #     target_position_read[2]=0.08

                #     target_pos = target_position_read.copy()
                #     target_pos[2]+=0.75
                # print(extend_dist)

#                     if target_arm[1] < 0:
#                         ee_body_name = "right_wrist_yaw_link"
#                         shoulder_body_name = "right_shoulder_pitch_link"
#                     else:
#                         ee_body_name = "left_wrist_yaw_link"
#                         shoulder_body_name = "left_shoulder_pitch_link"

#                     ee_pos = d.xpos[m.body(ee_body_name).id].copy()
#                     shoulder_pos = d.xpos[m.body(shoulder_body_name).id].copy()

#                 # =========================
#                 # 目标位置
#                 # =========================
#                     target_pos_np = np.asarray(target_pos, dtype=np.float64).copy()

#                 # 如果你的 target_arm 是 [x, y, z] 世界坐标，就直接用。
#                 # 如果 target_arm 是相对坐标，需要先转成世界坐标再记录。

#                 # =========================
#                     # 骨盆、脚位置
#                 # =========================
#                     pelvis_pos = d.xpos[m.body("pelvis").id].copy()

#                     left_foot_pos = d.xpos[m.body("left_ankle_roll_link").id].copy()
#                     right_foot_pos = d.xpos[m.body("right_ankle_roll_link").id].copy()

#                 # 如果你的模型脚底 body 不是这个名字，可以换成：
# # left_foot_pos = d.xpos[m.body("left_foot_link").id].copy()
# # right_foot_pos = d.xpos[m.body("right_foot_link").id].copy()

# # =========================
# # 腰部 pitch
# # =========================
#                     waist_pitch_idx = 14
#                     waist_pitch = d.qpos[7 + waist_pitch_idx]

# # =========================
# # action
# # =========================
# # 建议记录最终送给 PD 或 MuJoCo 的 action
# # 如果你有 action_interp，就用 action_interp
#                     current_action_for_metric = target_dof_pos.copy()

#                     wcgs_evaluator.record(
#     ee_pos=ee_pos,
#     target_pos=target_pos_np,
#     shoulder_pos=shoulder_pos,
#     pelvis_pos=pelvis_pos,
#     left_foot_pos=left_foot_pos,
#     right_foot_pos=right_foot_pos,
#     action=current_action_for_metric,
#     waist_pitch=waist_pitch,)
#                 else :        
#                         result = wcgs_evaluator.compute()

#                         result["target_x"] = float(target_arm[0])
#                         result["target_y"] = float(target_arm[1])
#                         result["target_z"] = float(target_arm[2])

#                         print("========== WCGS Result ==========")
#                         for k, v in result.items():
#                              print(f"{k}: {v}")

#                         save_wcgs_result_to_csv(csv_path, result)
                single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints,extend_dist,counter)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # Transform action to target_dof_pos
                target_dof_pos = action[:15] * config['action_scale'] + config['default_angles']
            
            # Sync viewer
            viewer.sync()
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
 
    main()
