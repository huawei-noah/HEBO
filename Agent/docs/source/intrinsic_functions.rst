.. _intrinsic-function-guide:


Intrinsic Functions
=====================================================

This guide will walk you through intrinsic functions and the `Command`:code: class.

Intrinsic functions should define a specific operation that affect the agent's memory.


The Command Class
------------------------

Intrinsic functions should inherit from `Command`:code:. This class will serve as the blueprint for customs functions.

.. code-block:: python

    from agent.commands.core import Command

    class CustomThinkCommand(Command):
        """A custom command to perform a specific task."""


Command Metadata
-----------------------

The command requires metadata, including its name and description. These serve an important role when the language model must make decisions about which commands to execute.
This leads to the following:

.. code-block:: python

    class CustomThinkCommand(Command):
        name: str = "think"
        description: str = "Produce a thought about the observation of the environment"

Specifically, the name and description of a command will be fed into the language model during decision flows. Here is a basic example of how these can be used to prompt the language model.

.. code-block::

  Your response should be an available command in the format "Command: <command>",
  where <command> is the name of the command below.
  Here is what happened so far:
    ...

  Here is a list of the available commands:
  - {name1}: {description1}.
  - {name2}: {description2}.

  Please pick between {name1} and {name2}.

Obviously, the bracketed terms are replaced with the names and descriptions of the commands as defined in this class.


Input and Output Keys
-----------------------------

The expected inputs and the outputs of the command are declared as class attributes using dictionaries.

- `input_keys`:code: :

  Memory keys that are expected to be in the memory when the function is executed.
  The `key`:code: to the dictionary is the argument name and the `value`:code: is passed as an argument value.
  For example, a dictionary such as `{"key": "value"}`:code: means that func will be called as `func(key=value)`:code:.
  The keyword arguments are collected from all of `input_keys`:code:, `output_keys`:code:, and `required_prompt_templates`:code:.
  For this reason, keys have to be unique among `input_keys`:code:, `output_keys`:code: and `required_prompt_templates`:code:.

- `output_keys`:code: :

  Memory keys that are going to be created or altered when the function is executed.
  The `key`:code: to the dictionary is the argument name and the `value`:code: is passed as an argument value.
  See documentation of `input_keys`:code: for an explanation of how this is used.

.. code-block:: python

    class CustomThinkCommand(Command):
        ...

        input_keys: Dict[str, MemKey] = {
            'observation_mem_key': MemKey.OBSERVATION,
        }
        output_keys: Dict[str, MemKey] = {
            'output_mem_key': MemKey.THOUGHT,
        }


Required Prompt Templates
--------------------------------

If a command relies on language model prompts, it should define the necessary templates. These are specified in the `required_prompt_templates`:code: dictionary.

- `required_prompt_templates`:code: :

  Prompt templates used by this function. These templates need to exist within the searchable paths.
  The `key`:code: to the dictionary is the argument name and the `value`:code: is passed as an argument value.
  See documentation of `input_keys`:code: for an explanation of how this is used.

.. code-block:: python

    class CustomThinkCommand(Command):
        ...

        required_prompt_templates: Dict[str, str] = {
            'ask_for_thought_template': 'think.jinja'
        }

The `func`:code: Method
-------------------------

The `func`:code: method is overwritten from the parent `Command`:code: class and defines the operation of the command. This method should accept `agent`:code: as its first parameter, followed by the required inputs defined in `input_keys`:code:, `output_keys`:code:, and `required_prompt_templates`:code:.
Note how the arguments to this function match the keys of the dictionaries provided above (input/output keys).

In this function, feel free to use `agent.memory`:code: for memory operations and `agent.llm`:code: for language model interactions, as well as using `agent.prompt_builder`:code: as shown below.

.. note::
  Right now there are no checks to enforce the use of input/output/templates that were defined above (i.e., you can do whatever you want here). However do *NOT* do this, as we do have plans of enforcing this in the future.
  Also, you might find that your functions are going to be easier to reuse or override if you stick to this template.

Here is an example of the `func()`:code: :

.. code-block:: python

    class CustomThinkCommand(Command):
        ...

        def func(
        self,
        agent,
        # see how the arguments below match the keys in `input_keys`, `output_keys`, and `required_prompt_templates`:
        observation_mem_key,
        output_mem_key,
        ask_for_thought_template
        ):

            # Example: Retrieve input from memory
            observation = agent.memory.retrieve({observation_mem_key: 1.0})

            # Example: Generate a prompt using a template
            prompt = agent.prompt_builder(ask_for_thought_template, {'text_obs': observation}) # key will be inserted in jinja template

            # Example: Call the language model and process the response
            response = agent.llm.chat_completion(prompt)

            # Example: Store the result in memory
            agent.memory.store(response, {output_mem_key})


Shortcuts to Delete and Rename Memory Keys (Optional)
-----------------------------------------------------

Optionally, it is possible to define memory keys that should be renamed or deleted after the command's execution using `renamed_keys`:code: and `deleted_keys`:code:.
These are shortcuts and keys can always be deleted or renamed without including them in the `output_keys`:code: dictionary.

- `renamed_keys`:code: :

  Memory keys that will be renamed. The key of this dictionary is the source name and the value is the new name.
  Renaming happens after the main function is executed, but before deletions.

- `deleted_keys`:code: :

  Memory keys that will be deleted. Deletion happens at the end of the function call.

.. code-block:: python

    class CustomThinkCommand(Command):
        ...

        renamed_keys: Dict[MemKey, MemKey] = {
            MemKey.OLD_KEY: MemKey.NEW_KEY
        }
        deleted_keys: List[MemKey] = [MemKey.KEY_TO_DELETE]
