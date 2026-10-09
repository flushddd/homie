import numpy as np
import pinocchio as pin


def build_g1_reduced_robot(urdf_path: str, mesh_dir: str):
    robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

    mixed_jointsToLockIDs = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",

        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",

        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]

    reduced_robot = robot.buildReducedRobot(
        list_of_joints_to_lock=mixed_jointsToLockIDs,
        reference_configuration=np.zeros(robot.model.nq),
    )

    # add L_ee / R_ee frames (same as your IK code)
    reduced_robot.model.addFrame(
        pin.Frame(
            "L_ee",
            reduced_robot.model.getJointId("left_wrist_yaw_joint"),
            pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    reduced_robot.model.addFrame(
        pin.Frame(
            "R_ee",
            reduced_robot.model.getJointId("right_wrist_yaw_joint"),
            pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )

    # ✅ VERY IMPORTANT: model changed => recreate data
    reduced_robot.data = reduced_robot.model.createData()

    return reduced_robot


def sample_random_q_in_limits(model, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    lb = np.array(model.lowerPositionLimit).copy()
    ub = np.array(model.upperPositionLimit).copy()

    inf_mask = ~np.isfinite(lb) | ~np.isfinite(ub)
    lb[inf_mask] = -np.pi
    ub[inf_mask] =  np.pi

    return rng.uniform(lb, ub)


def fk_ee_poses(reduced_robot, q: np.ndarray):
    model = reduced_robot.model

    # ✅ safest: create fresh data each call (optional but robust)
    data = model.createData()

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    L_id = model.getFrameId("L_ee")
    R_id = model.getFrameId("R_ee")

    # getFrameId 有的版本找不到会返回 model.nframes（越界）
    if L_id >= model.nframes or R_id >= model.nframes:
        raise RuntimeError(f"Frame not found: L_id={L_id}, R_id={R_id}, nframes={model.nframes}")

    oT_L = data.oMf[L_id]
    oT_R = data.oMf[R_id]
    return oT_L, oT_R


if __name__ == "__main__":
    urdf_path = "/home/eisr/avp_teleoperate/assets/g1/g1_body29_hand14.urdf"
    mesh_dir  = "/home/eisr/avp_teleoperate/assets/g1"

    rr = build_g1_reduced_robot(urdf_path, mesh_dir)

    q = sample_random_q_in_limits(rr.model)
    oT_L, oT_R = fk_ee_poses(rr, q)

    np.set_printoptions(precision=5, suppress=True)
    print("Random q:\n", q)

    print("\nL_ee pose:")
    print("R=\n", oT_L.rotation)
    print("t=\n", oT_L.translation)
    print("4x4=\n", oT_L.homogeneous)

    print("\nR_ee pose:")
    print("R=\n", oT_R.rotation)
    print("t=\n", oT_R.translation)
    print("4x4=\n", oT_R.homogeneous)
