cmake_minimum_required(VERSION 3.0.2)
project(observation_manager)

find_package(catkin REQUIRED COMPONENTS
  rosllm_srvs
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/observation_manager_node
  test/test_sensor_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)