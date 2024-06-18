# ROS-LLM

The ROS-LLM framework.

# Packages

* `agent_comm`: communication interface with the AI agent.
* `behavior_exector`: contains nodes for executing behaviours in ROS produced by an LLM.
* `rosllm_msgs`: common message types used by the ROSLLM framework.
* `rosllm_srvs`: common service types used by the ROSLLM framework.
* External packages are found in the `extern` directory:
  * `BehaviorTree/BehaviorTree.ROS`: `BehaviorTree.CPP` utilities to work with ROS.
  * `ros/executive_smach`: A procedural python-based task execution framework with ROS integration.

# Install

Ensure [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) is installed before.
Then go through the following steps:
1. Ensure ROS Noetic is sourced: `source /opt/ros/noetic/setup.bash`
2. Create catkin workspace: `mkdir -p rosllm_ws/src`
3. Initialize workspace: `cd rosllm_ws; catkin init`
4. Clone rosllm: `cd src; git clone --recursive REPO` where `REPO` is the repository link.
5. Install dependancies: `rosdep install -i -r -y --from-paths . --ignore-src`
6. Build workspace: `catkin build -s`

# Support

Currently, only ROS Noetic is supported.
Support for ROS2 is planned for the future.