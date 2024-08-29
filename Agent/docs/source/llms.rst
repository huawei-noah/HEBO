.. _llms:


LLMs
=====================================================

LLMs are the backbone of agents within Agent.
They are charged with the core reasoning and decision-making processes.
Agent supports a wide range of LLMs and provides a flexible interface for users to define their own LLMs.

There are four main types of LLMs compatible with Agent:

- from HuggingFace hub or a local file
- from a FastChat server
- from OpenAI (GPT models)
- a 'human' stdin interface

These are supported by corresponding backends in the framework, which all share a common interface.

Core Interface
---------------
The `LanguageBackend`:code: abstract class is the core interface for all LLMs.
It provides a common interface, making it easy to switch between LLMs.
It defines common class attributes, such as the model id, logger, context length, and history of prompts.
It also introduces the following core methods which all backends must override:

- `count_tokens`:code: : Counts the number of tokens in a given prompt, using the model's tokenization method.
- `chat_completion`:code: : Generates an answer to a prompt given in a chat-like interaction.
- `choose_from_options`:code: : Asks the model to choose from a list of options, based on the given prompt.


Different Backends
-----------------------
Several backends enable Agent to offer support for a wide range of different LLM types and hosting methods.
We note the following core backends:

- **HuggingFace backend**: Leverages the HuggingFace transformers library to load models from the HuggingFace model hub or from a local file. This enables users to use any models hosted on HuggingFace, or which they have downloaded or trained locally.

- **FastChat backend**: Connects to a FastChat server to use a model hosted there. This is useful for using models which are too large to run on a local machine, or which are hosted on a remote server. It uses the OpenAI API to communicate with the FastChat server.

- **OpenAI backend**: Connects to the OpenAI API to use OpenAI models. This is useful for using OpenAI's GPT models, which are not open-source and available on HuggingFace.

- **Human backend**: A simple backend which allows a human user to provide responses. The user will be prompted for an input, which will be used as the 'LLM' output. This is useful for testing and debugging various agent interactions without needing to use a model.

You can create your own backend by subclassing `LanguageBackend`:code: and implementing the required methods, if you encounter a scenario not covered by one of the above backends.


LLM Configs
---------
Whenever you want to use an LLM, you will need to provide a yaml config file under `configs/llm/`:code:.
This config file will specify the backend to use, and any other relevant parameters for model.
All backends require a config with the following fields:

- `model_id`:code: : The model id to use. This will be the model name for HuggingFace models, the path to the model for local models, or the model id for FastChat and OpenAI models.
- `context_length`:code: : The LLM's maximum context length, used to handle out-of-context errors.
- `_target_`:code: The backend to use. e.g `agent.models.HuggingFaceLanguageBackend`:code:.
- `_partial_`:code: This should always be set to 'true' and ensure hydra instantiation works as needed.

In addition to these mandatory fields, each backend may require additional ones.
For example, the HuggingFace backend requires `model_kwargs`:code: and `tokenizer_kwargs`:code: fields, while the FastChat and OpenAI backends require `server_ip`:code: and `api_key`:code: fields.

For ease of use, we provide pre-defined configs for each of the backends, respectively named `hf`:code:, `fschat`:code:, `openai`:code:, and `human`:code:.
These can simply be overridden with the required custom fields (`model_id`:code: and `context_length`:code:).
An example config for OpenChat-3.5 is shown below:

.. literalinclude:: ../../configs/llm/hf/openchat_3.5.yaml
   :language: yaml


Custom LLMs
----------------
You can easily add your own LLM to Agent through the config files.
Please see the `Adding a New LLM <https://gitlab-uk.rnd.huawei.com/ai-uk-team/reinforcement_learning_london/agent/agent/-/tree/main/tutorials/create_new_llm.ipynb/>`_ tutorial for a guided example on using your own LLMs within Agent.


Setting Up a FastChat Server
-------------------------------

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

After running these commands, the user has to modify the server_ip argument of the fschat.yaml file:

   .. code-block:: yaml

      server_ip: http://server_ip_controller:8000/v1
