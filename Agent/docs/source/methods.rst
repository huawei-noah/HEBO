.. _methods:


Methods
=====================================================

Methods define prompting mechanisms used to query the agent in Agent.
Accompanied by prompt templates, they define the sequence of prompts submitted to the LLM.
They are responsible for the agent's 'strategy', defining the series of Commandsintrinsic functions the agent should execute before taking actions in a given task.

:doc:`flows` and :doc:`intrinsic_functions` form a core part of methods, and are essentially the building blocks.
As such, it is strongly recommended to read the :doc:`flows` and :doc:`intrinsic_functions` documentation along with this page, especially if you are interested in building your own methods.


Method Structure
-----------------
Methods define the agent's high-level flows.
There are several of these, which define all actions the agent should take, or at least consider, at different stages.
We explain the following high-level agent flows:

- `run_on_init_flow()`:code: : Defines any actions the agent should take upon initialization, before the first episode even starts.
- `run_on_episode_start_flow()`:code: : Defines any actions the agent should take at the start of each episode.
- `run_pre_action_flow()`:code: : Defines any actions the agent should take before submitting an action to the environment.
- `run_post_action_flow()`:code: : Defines any actions the agent should take after submitting an action to the environment.
- `run_on_episode_end_flow()`:code: : Defines any actions the agent should take at the end of each episode.

Each of these can be as complex as required, made up of multiple flows and intrinsic functions of various types.

Most methods will likely only need to define the `run_pre_action_flow()`:code:, which is actually charged with generating the action to submit to the environment.
In fact, `run_pre_action_flow()`:code: is the only one which **must** be defined by a new method, the others are optional.
Moreover, `run_pre_action_flow()`:code: must include the `Act`:code: extrinsic function (or a composite command including `Act`:code:) to ensure the agent submits an action to the environment.


Method Configs
---------------------
While they depend on defined flows and intrinsic functions, methods themselves are defined exclusively through a yaml configuration file, under `configs/method/`:code:.
The configuration filename should match the method name, and the configuration defines the various high-level flows mentioned above, as well as any method-specifc variables the prompt builder should have access to.
An example config for the `direct`:code: method is given below:

.. literalinclude:: ../../configs/method/direct.yaml
   :language: yaml

Here, the method defines only the `agent.pre_action_flow`:code:, composed of a single seuqnetial flow with a single `Act`:code: command.
This is the bare minimum a method should define.
It also defines `agent.prompt_builder.default_kwargs.cot_type`:code:, which sets a variable the prompt builder will be able to use to decide whether to include a request for chain-of-thought (CoT) in the prompt.


Existing Methods
--------------------
Agent comes packaged with several core inbuilt reasoning methods. The core methods included are:

- `direct`:code: : Prompts the agent to directly output an action for the environment.
- `zs-cot`:code: : Prompts the agent to generate thoughts before giving an action.
- `fs`:code: : Prompts the agent to directly output an action for the environment, with in-context examples provided in the prompt.
- `fs-cot`:code: : Prompts the agent to generate thoughts before giving an action, with in-context examples provided in the prompt.
- `fs-cot-react`:code: : Prompts the agent to first only generate a thought, and then an action in the subsequent reasoning step.
- `fs-cot-reflect`:code: : Prompts the agent to first reflect on past trajectory, and then generate an action in the subsequent reasoning step.
- `fs-cot-zerostep-reflect`:code: : Prompts the agent to give an action, and then immediately reflect on whether that action is the best choice.
- `fs-cot-sc`:code: : Runs fs-cot several times and selects most consistent action.
- `fs-least2most`:code: : Prompts the agent to decompose the problem into sub-problems, and then to solve the subproblems before generating an action.

Most methods are available for all types of tasks. However, a few are specific to single-step or multi-step tasks.
`fs-cot-reflect`:code: reflects on past trajectory and is therefore only suitable for multi-step tasks, while `fs-cot-zerostep-reflect`:code: is the approximate equivalent for single-step tasks.
Furthermore, `fs-least2most`:code: is only suitable for single-step tasks, requiring all subproblems to be solved in a single step.


Custom Methods
---------------
In addition to all existing methods, you can easily create your own methods by writing new method configuration file.
If you require new :doc:`flows` or :doc:`intrinsic_functions`, these should also be defined in the appropriate fashion.

Please see the `Creating a New Method <https://gitlab-uk.rnd.huawei.com/ai-uk-team/reinforcement_learning_london/agent/agent/-/tree/main/tutorials/create_new_method.ipynb/>`_ tutorial for a guided example on creating and using a new method.
