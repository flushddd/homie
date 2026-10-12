# Analogous to kuavo_msgs/jointCmd: controller → sim (or HW).
# control_mode: 0=PD position (tau = kp*(q_des-q) - kd*dq), 1=torque
std_msgs/Header header
string[] name
float64[] joint_q
float64[] joint_v
float64[] tau
float64[] joint_kp
float64[] joint_kd
int32[] control_mode
