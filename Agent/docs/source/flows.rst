
Flows
=============


In the architecture of Agent, "flows" serve as a mechanism for guiding the agent's interactions and decision-making processes
within its environment. These flows are designed to systematically organize and manage the various intrinsic and extrinsic functions that an
agent utilizes to process information, analyze situations, and determine actions.
These intrinsic functions include activities such as thinking, querying databases, breaking down complex tasks into manageable components, etc.
There are three primary types of flows structures:
#. Sequential flows
#. Decision Flows
#. Loop flows.

Sequential flows are straightforward, linear progressions of intrinsic functions, enabling a clear execution path for single or multiple tasks.
Decision flows allow the agent to select between different sub-flows or intrinsic/extrinsic function.
Loop flows offer a mechanism for repeating a particular sequence of functions for a set number of iterations.

By employing these diverse flow structures, users can precisely configure the agent's approach to problem-solving and task execution,
enhancing its efficiency and effectiveness in dynamic environments. This modular and flexible framework is integral for
developing a sophisticated AI Agent.

Agent Flows
--------------------

The agent by default has **five** flows.

#. ``on_init_flow`` This flow is executed after the agent initialisation. It is responsible for initialising any external tool that the agent may need. This flow is optional and can be None.
#. ``on_episode_start_flow`` This flow is executed after the environment gets reset. It is used to initialise any information that the agent may need during the episode.  This flow is optional and can be None.
#. ``pre_action_flow`` This flow is executed to allow estimating the action that the agent will perform. This flow is **mandatory** and **must** terminate with the command ``Act()``.
#. ``post_action_flow`` This flow is executed after the agent performs the action in the environment. It can be used to process the newly perceived observation and reward or reflect on the outcome of the action. This flow is optional and can be None.
#. ``on_episode_end`` This flow is executed after the episode terminates. It can be used to post-process the episode that terminated. For example, the episode can be transformed to a trajectory that will be used to train the agent. This flow is optional and can be None.


Flow Structures
--------------------
Agent contain three basic structures of flows

``SequentialFlow`` This flow is used to define a list of commands that will be executed in a sequential order

.. code-block:: python

    class SequentialFlow(Flow):
    """A Flow subclass that represents a sequential execution of commands or sub-flows.

    Args:
        sequence (list): A list of commands or sub-flows to be executed in order.
        name (str): The name of the flow. Defaults to "sequential_flow".
        description (str): A brief description of the flow. Defaults to "A sequence of actions".
    """


``DecisionFlow`` This flow is used to define a selection between different commands or sub-flows. The llm is asked to decide between different choices.

.. code-block:: python

    class DecisionFlow(Flow):
    """A Flow subclass representing a decision point where one command or sub-flow is chosen from
    multiple choices using the agent's LLM (Language Model).

    The LLM is prompted using a template (`flow_select.jinja`) to make a choice
    among the available commands or sub-flows. By default, the template makes use
    of the "name" and "description" attributes of sub-flows/commands.

   Args:
        choices (list): A list of commands or sub-flows from which one will be chosen and executed.
        name (str): The name of the flow. Defaults to "decision_flow".
        description (str): A brief description of the flow. Defaults to "Decide between choices".
        """


``LoopFlow`` This flow is used to allow looping through a command or sub-flow

.. code-block:: python

    class LoopFlow:
    """A Flow subclass that represents a loop execution of commands or sub-flows that repeats
    itself at most max_repetitions times.

    Args:
        loop_body: A flow or a command that will be repeated a number of times.
        max_repetitions: The maximum number of times that the sequence will be repeated.
        allow_early_break: If true, ask the LLM at the end of the sequence whether to break from the loop.
        name: The name of the flow. Defaults to "sequential_flow".
        description: A brief description of the flow. Defaults to "A sequence of actions".
    """



Simplest Flow
--------------------

Let's know design the simplest possible flow. This flow will consist of a single-command. We will design an agent that only call the ``Act()`` command to select the action that will execute.
More specifically we design the ``pre_action_flow``

.. code-block:: python

    from agent.commands import Act
    from agent.commands import SequentialFlow
    # assume the agent is initialised
    # initialise the flow for the agent
    agent.pre_action_flow = SequentialFlow([Act()])
    # run the flow
    agent.run_pre_action_flow()

This can be implemented through the hydra configuration using the following config file

.. code-block:: yaml

    # @package _global_
    agent:
        pre_action_flow:
            _target_: agent.commands.SequentialFlow
            sequence:
            - _target_: agent.commands.Act


Nesting Commands (Basic)
------------------------

It is straight forward to nest commands and flows to create structures of reasoning. For example, we will now go through the implementation
of the ReAct method. In this method, the agent either has to think and execute an action or directly execute an action.
ReAct will be implemented using a decision and a sequential flow.

.. code-block:: python

    from agent.commands import Act, Think
    from agent.commands import SequentialFlow, DecisionFlow
    # assume the agent is initialised
    # initialise the flow for the agent
    choice1 = SequentialFlow([Think(), Act()])
    choice2 = Act()
    agent.pre_action_flow = Decision([choice1, choice2])
    # run the flow
    agent.run_pre_action_flow()

This can also be implemented in the hydra configuration using the following config:

.. code-block:: yaml

    # @package _global_
    agent:
        pre_action_flow:
            _target_: agent.commands.DecisionFlow
            choices:
            - _target_: agent.commands.SequentialFlow
                name: react
                description: Think and then act
                sequence:
                - _target_: agent.commands.Think
                - _target_: agent.commands.Act
            - _target_: agent.commands.Act
