cmake_minimum_required(VERSION 3.0.2)
project(rosllm_srvs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
  message_runtime
)

add_service_files(
  FILES
  ExecuteBehavior.srv
  AtomicAction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
