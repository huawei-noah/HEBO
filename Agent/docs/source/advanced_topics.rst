.. _advanced-topics:

Advanced Topics
===============

In this section, we delve into more advanced aspects of the Agent framework, including server setup, training procedures, and troubleshooting common issues like proxy problems.

Server Setup
------------

.. _Server Setup:



Setting Up a FastChat Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Agent framework, the agents utilise LLMs through the OpenAI API, using the openai.chat.completion.create function.
Using the GPT models can be automatically done by adding the OPENAI_API_KEY in the .env file as mentioned above.
However, Agent supports several open-source LLMs, such as Llama, Openchat, etc. To allow the Agent to utilise these LLMs, the user should create a server that hosts these models.
Some of the LLMs, such as Openchat-3.5 offer directly support to deploy such server. However, we recommend using a FastChat server.
Assuming a server with the IP address server_ip_controller will host the fastchat controller, one can initiate the fastchat controller on port 8000 using the following commands:


1. Initiate the FastChat controller:

   .. code-block:: bash

      python -m fastchat.serve.controller --host 0.0.0.0

2. Start the FastChat server:

   .. code-block:: bash

      python -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 8000

Afterward, users need to start the workers responsible for running the LLM. This can be on a different server with the IP address server_ip_worker, as long as the two servers can communicate. By default, the controller listens for LLM workers on port 21001. The vllm library is used to run the worker, and a vllm worker can be initiated on the server_ip_worker server at port 31020 with the following command:

Start the workers:

   .. code-block:: bash

      CUDA_VISIBLE_DEVICES=0 python -m fastchat.serve.vllm_worker --model-path openchat/openchat_3.5 --controller http://server_ip_controller:21001 --port 31020 --worker-address http://server_ip_worker:31020 --host 0.0.0.0 --dtype=float16 

After running these commands, the user has to modify the server_ip argument of the fschat.yaml file to

Modify the `fschat.yaml` file:

   .. code-block:: yaml

      server_ip: http://server_ip_controller:8000/v1

Training
--------

.. _Training:

Finetuning with Huggingface Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework supports finetuning with the Huggingface backend. To begin training:

.. code-block:: bash

   python src/agent/train_rlft.py

For multi-GPU training, use:

.. code-block:: bash

   accelerate launch --config_file configs/training/deepspeed/stage0.yaml src/agent/train_rlft.py
