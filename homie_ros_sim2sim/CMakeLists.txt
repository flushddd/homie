cmake_minimum_required(VERSION 3.0.2)
project(homie_ros_sim2sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  HomieJointData.msg
  HomieImuData.msg
  HomieSensorsData.msg
  HomieJointCmd.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    message_runtime
)

catkin_install_python(PROGRAMS
  scripts/mujoco_sim_node
  scripts/rl_policy_node
  scripts/keyboard_cmd_node
  scripts/joy_cmd_node
  scripts/export_him_onnx.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch policies
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
