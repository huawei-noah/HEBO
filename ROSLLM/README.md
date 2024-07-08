# ROS-LLM

The ROS-LLM framework.

Please see the following links
* [Paper (arXiv)](https://arxiv.org/abs/2406.19741)
* [Website](https://rosllm.github.io/)
* [Code](https://github.com/huawei-noah/HEBO/tree/master/ROSLLM)

# Packages

* `agent_comm`: communication interface with the AI agent.
* `behavior_executor`: contains nodes for executing behaviours in ROS produced by an LLM.
* `rosllm_msgs`: common message types used by the ROSLLM framework.
* `rosllm_srvs`: common service types used by the ROSLLM framework.
* External packages are found in the `extern` directory:
  * `BehaviorTree/BehaviorTree.ROS`: `BehaviorTree.CPP` utilities to work with ROS.
  * `ros/executive_smach`: A procedural python-based task execution framework with ROS integration.

# Install

First, please ensure ROS Noetic and [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html) are installed.
Then step through the following steps:
1. Open a new terminal and ensure ROS Noetic is sourced: `source /opt/ros/noetic/setup.bash`
2. Create catkin workspace: `mkdir -p rosllm_ws/src`
3. Change directory: `cd rosllm_ws`
4. Initialize workspace: `catkin init`
5. Clone HEBO:
  * **IMPORTANT**: make sure you *do not* clone the `HEBO` package in the `src` directory. It should be cloned into the `rosllm_ws/` directory.
  * **IMPORTANT**: also make sure to include the `--recursive` flag when cloning.
  * Clone via ssh: `git clone --recursive git@github.com:huawei-noah/HEBO.git`
  * Clone via https: `git clone --recursive https://github.com/huawei-noah/HEBO.git`
6. Change directory: `cd src`
7. Create a symbolic link: `ln -s ../HEBO/ROSLLM/`
8. Install dependancies:
  * `rosdep install -i -r -y --from-paths . --ignore-src`
  * `pip3 install openai PyYAML gradio`
9. Build workspace: `catkin build -s`

# Support

Currently, only ROS Noetic is supported.
Support for ROS2 is planned for the future.

# Citation

If you use the ROS-LLM framework in your work, please cite
```
@misc{mower2024rosllmrosframeworkembodied,
      title={ROS-LLM: A ROS framework for embodied AI with task feedback and structured reasoning},
      author={Christopher E. Mower and Yuhui Wan and Hongzhan Yu and Antoine Grosnit and Jonas Gonzalez-Billandon and Matthieu Zimmer and Jinlong Wang and Xinyu Zhang and Yao Zhao and Anbang Zhai and Puze Liu and Davide Tateo and Cesar Cadena and Marco Hutter and Jan Peters and Guangjian Tian and Yuzheng Zhuang and Kun Shao and Xingyue Quan and Jianye Hao and Jun Wang and Haitham Bou-Ammar},
      year={2024},
      eprint={2406.19741},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2406.19741},
}
```