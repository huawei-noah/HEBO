cmake_minimum_required(VERSION 3.0.2)
project(ros_agent)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

# catkin_python_setup()

add_service_files(
  FILES
  HandleAgentAction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)