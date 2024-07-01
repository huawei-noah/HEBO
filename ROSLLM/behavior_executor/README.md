# Behavior Executor

This package is responsible for executing behaviors in ROS.
Several nodes are implemented for executing different behavior representations.
Each node implements a service called `behavior_executor` with type `rosllm_srvs/ExecuteBehavior`.
The service has a string input defining the behaviour (this is assumed to be the output from an LLM).
A response from the service contains four pieces of information:
a success flag (true/false),
a human-readable message, and
an integer information flag.

Currently, the following nodes are implemented:
* `behavior_tree_executor`: input is a behaviour tree.
* `code_executor`: input is a python script containing a `main` function.
* `sequence_executor`: input is a sequence of ROS action/service names separated by newlines (i.e. `\n`).

# Examples

Checkout the tests in `test` directory.
For each executor, follow these steps:

1. In a new terminal, run: `roscore`
2. In a new terminal, run: `rosrun behavior_executor NAME_executor` where `NAME` specifies the executor type (e.g. `code`)
3. If you ran the `behavior_tree_executor` then in a new terminal run: `rosrun behaviortree_ros test_server`
4. In a new terminal, run `rosrun behavior_executor test_NAME_executor` where `NAME` is the same as in step 2.