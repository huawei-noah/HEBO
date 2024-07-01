# Observation manager

This package provides tools for handling observations from various sensors used in your setup.
The main use is by implementing several `rosllm_srvs/Observation` services that when called provide an observation of the environment.
Alongside these, a central node is launched called `observation_manager_node` that setups a connection to all these sub-services.
The manager allows you to make a single call to the `get_observation` service.

# Example

Open a terminal and run
```
$ roslaunch observation_manager test.launch
```

In a second terminal, run
```
$ rosservice call /get_observation "{}"
```