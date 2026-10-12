# homie_ros_sim2sim

ROS1 (Noetic) **sim2sim** stack for Homie / Unitree G1, modelled on
[`kuavo_ros_control`](https://github.com/) `mujoco_cpp` ↔ controller topic bridge.

## Architecture

```
┌──────────────────┐   /homie/sensors_data    ┌──────────────────┐
│  mujoco_sim_node │ ───────────────────────► │  rl_policy_node  │
│  (physics + PD)  │ ◄─────────────────────── │  (TorchScript)   │
└──────────────────┘   /homie/joint_cmd       └────────┬─────────┘
                                                       ▲
                              /cmd_vel, /homie/height_cmd
```

Same topic contract can later swap `mujoco_sim_node` for a real-robot bridge
(Unitree SDK / DDS), without changing the policy node — the kuavo pattern.

| Topic | Type | Direction |
|-------|------|-----------|
| `/homie/sensors_data` | `HomieSensorsData` | sim → policy |
| `/homie/joint_cmd` | `HomieJointCmd` | policy → sim |
| `/homie/joint_states` | `sensor_msgs/JointState` | debug |
| `/cmd_vel` | `geometry_msgs/Twist` | teleop → policy |
| `/homie/height_cmd` | `std_msgs/Float64` | teleop → policy |

## Build

```bash
# one-time: overlay into a catkin ws
mkdir -p ~/homie_ws/src
ln -sfn ~/OpenHomie/homie_ros_sim2sim ~/homie_ws/src/homie_ros_sim2sim
cd ~/homie_ws
source /opt/ros/noetic/setup.bash
# Prefer system Python 3.8 (Noetic); avoid conda/empy conflicts
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

Python deps (mjlab venv): `mujoco`, `torch`, `onnxruntime`, `numpy`.
Policy runtime for sim2sim/deploy is **ONNX** (same as `HomieDeploy` / kuavo); `.pt` is only for export/debug.

## Run

### Export ONNX (recommended, same path as HomieDeploy)

```bash
# from mjlab venv
python $(rospack find homie_ros_sim2sim)/scripts/export_him_onnx.py \
  --checkpoint $HOME/OpenHomie/HomieRL_mjlab/logs/rsl_rl/homie_g1/2026-10-10_16-30-36/model_11000.pt \
  --output $(rospack find homie_ros_sim2sim)/policies/g1_phase1.onnx \
  --num-actions 12 --one-step 80 --hist-len 6
```

Phase-1 (12-DOF legs, ONNX):

```bash
source /opt/ros/noetic/setup.bash
source ~/homie_ws/devel/setup.bash
roslaunch homie_ros_sim2sim sim2sim_g1.launch \
  phase:=1 \
  phase_config:=$(rospack find homie_ros_sim2sim)/config/g1_phase1.yaml \
  policy_path:=$(rospack find homie_ros_sim2sim)/policies/g1_phase1.onnx
```

Phase-2 (14-DOF legs + waist roll/pitch, yaw held):

```bash
roslaunch homie_ros_sim2sim sim2sim_g1.launch \
  phase:=2 \
  phase_config:=$(rospack find homie_ros_sim2sim)/config/g1_phase2.yaml \
  policy_path:=/path/to/phase2_policy.pt
```

**Keyboard teleop is on by default** (`keyboard:=true`): focus the small pygame window.

| Key | Action |
|-----|--------|
| w/s | forward / back |
| a/d | left / right |
| q/e | yaw |
| r/f | height |
| space | stop |

Gamepad (北通 A1N3): `joystick:=true keyboard:=false`. Mapping in `config/joy_teleop.yaml`.

## Config

- `config/g1_common.yaml` — joint names, default pose, PD gains
- `config/g1_phase1.yaml` / `g1_phase2.yaml` — action set, obs scales, flags

Obs layout matches `MujocoDeploy/mujoco_deploy_g1.py` (history × single frame).

## Relation to existing code

| Package | Role |
|---------|------|
| `MujocoDeploy/` | Monolithic Python sim2sim (no ROS) |
| `homie_ros_sim2sim/` | Split ROS nodes + msgs (this package) |
| `HomieDeploy/` | LCM / Unitree SDK real deploy |
| kuavo `mujoco_cpp` | C++ MuJoCo + LCM/ROS reference architecture |
