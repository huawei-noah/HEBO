# Agent

This library allows you to run many tasks using LLMs in a modular way!

## Documentation

You can compile the documentation yourself under `docs/` by running:

```bash
cd docs
make html
```

There are also several helpful [tutorials](tutorials/) to help you get started with running and customizing Agent.


## ROS interface

To run the agent with Flask interface with ROS 

In conda environment using the following command:

```
python ../src/agent/start.py task=ros_task method=direct llm@agent.llm=human
```

Replace the ``llm@agent.llm=human`` with your actual model.

In ROS environment follow the instruction in the ros_pange_agent packge
