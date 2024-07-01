Reinforcement Learning (WIP)
==========================

We are currently implementing a new unified version of RL for Agent.
Our implementations and results currently are limited and will be extended over the next months.
It is currently not merged with the main branch, and you have to checkout in the cleanup-train branch.

.. code-block:: bash

    git checkout cleanup-train


Basic implementation
----------------------

We have implemented the training loop of RL as an external Redis server.
Consider a client-server implementation, where the server hosts the LLM both for training and inference, while the client consists of different processes of the `agent`, where each the agent interacts with one or more environments.
Our default implementation currently is the A2C algorithm. To start the redis server for training in GPUs 0 to 3, using the Openchat-3.5 LLM, run the following command:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch src/agent/train.py llm=training/openchat_3.5

Now, we need to run the client which will run the `agent`. The client is responsible to interact with the environment and gather trajectories. To start 4 runs of `agent` in the babyai task, run the following command:

.. code-block:: bash

    python src/agent/start.py task=babyai_chat method=direct llm@agent.llm=distributed_hf agent.llm.model_id=openchat/openchat_3.5 agent=train +seed=0,1,2,3 hydra=parallel experiment_name=test -m
