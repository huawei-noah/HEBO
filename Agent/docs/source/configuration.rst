
Configuration Overview
=============

This section covers the details of configuring the Agent Framework. The framework uses Hydra for configuration management, allowing flexibility and ease in setting up different components like Agents, LLMs, Methods, and Tasks.

Hydra Overview
--------------

Hydra is a powerful tool for configuring complex applications. It simplifies the process of managing configurations, enabling users to compose configurations dynamically.

.. tip::
   For more information on Hydra, visit the official documentation: https://hydra.cc/docs/intro/

Agent Configuration
-------------------

Agent configuration files define the default settings for Hydra and link to specific command classes, memory management, and prompt building settings.

.. code-block:: yaml

   # Example agent configuration
   ├── agent/
   │   ├── commands
   │   │    └── default.yaml
   │   ├── memory
   │   │    └── default.yaml
   │   └── prompt_builder
   │        └── default.yaml

LLM Configuration
-----------------

LLM (Large Language Model) configuration files specify settings for various language models. These files include parameters such as model IDs, API keys, and server information.

.. tabs::

   .. group-tab:: GPT-3.5

      .. code-block:: yaml

         # Example LLM configuration for GPT-3.5
         _target_: src.agent.models.openai_api.OpenAIBackend
         _partial_: true
         server_ip: https://api.openai.com/v1
         api_key: ${oc.env:OPENAI_API_KEY}
         model_id: gpt-3.5-turbo
         context_length: 4096

   .. group-tab:: OpenChat

      .. code-block:: yaml

         # Example LLM configuration for OpenChat
         _target_: src.agent.models.fastchat_api.FastChatAPILanguageBackend
         _partial_: true
         model_id: openchat_3.5
         server_ip: http://127.0.0.1:8000/v1
         api_key: EMPTY
         context_length: 4096

Method Configuration
--------------------

Method configuration files determine the approach and strategy the agent uses in processing and responding to tasks. This includes selecting prompting methods and setting up the flow of commands.

.. code-block:: yaml

   # Example method configuration
   agent:
      pre_action_flow:
         _target_: agent.commands.SequentialFlow
         sequence:
            - _target_: agent.commands.Act
      prompt_builder:
         default_kwargs:
            cot_type: zero_shot

Task Configuration
------------------

Task configurations define specific settings for different tasks, including environment-specific variables, training parameters, and prompt templates.

.. code-block:: yaml

   # Example task configuration for Alfworld
   ├── task/
   │   ├── alfworld.yaml
   │   ├── babyai.yaml
   │   └── webshop.yaml

.. note::
   Task-specific configuration files often refer to the original papers or documentation for each task for detailed settings.

Environment Variables
---------------------

Some configurations require setting up environment variables, such as API keys for LLMs. These can typically be stored in a `.env` file in the project root.

.. code-block:: bash

   # Example .env file
   OPENAI_API_KEY='your_api_key_here'

Conclusion
----------

Proper configuration is crucial for the flexible and effective use of the Agent Framework. By leveraging Hydra and structured configuration files
