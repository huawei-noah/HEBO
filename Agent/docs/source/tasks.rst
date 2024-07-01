.. _tasks:


Tasks
=====================================================

Tasks are the means by which agents within Agent can be evaluated.
Agents interact with the tasks to make observations, take actions, and receive rewards.
Agent defines 6 tasks for fair evaluation of agents, as well as the ability to define custom tasks.


Task Interface
-----------------
All tasks within Agent are subclasses of the `Task`:code: class, which provides a common interface for agents to interact with tasks.
The `Task`:code: class defines a couple class attributes and some abstract methods as follows:

.. literalinclude:: ../../src/agent/tasks/tasks.py
   :pyobject: Task

This resembles the gym interface for tasks with its reset and step methods.
We additionally require an `answer_parser`:code: method which provides insstruction about how to extract the action from the raw LLM answer for a given task.
The `action_space`:code: class attribute defines whether the task's action space is discrete or continuous and is required for some :doc:`methods` to work properly.
Finally, the `task_id`:code: attribute is optional and can be used to identify the current task in the logs.


Task Configs
-----------------
All tasks much also have a corresponding task configuration file under `configs/task/`:code:, which defines the task's name, class, and any additional parameters required for the task to run.
We show the GSM8k config file to help illustrate this:

.. literalinclude:: ../../configs/task/gsm8k.yaml
   :language: yaml

The following fields are mandatory for any task:

- `task._target_`:code: : The task class (overriding the aforementioned interface).
- `agent.prompt_builder.template_paths`:code: : A list of folder names within `src/agent/prompts/templates/`:code: that contain the prompt templates for the task. The order matters, as they will define the priority of templates (e.g. if there is a template with the same name in two folders, the one in the first folder will be used).

The following fields are optional for any task:

- `max_episodes`:code: : The maximum number of episodes to run the task for. If not provided, the task will run indefinitely.
- `max_env_steps`:code: : The maximum number of task steps to run each episode for. Most tasks will probably already define this number, but this can help reduce the upper bound.

In addition to these fields available for all tasks, each task can define any number of additional fields under `task`:code: in the config file, such as `split`:code: in the example above.
These will be passed to the task class defined under `task._target_`:code: as keyword arguments when the task is created.


Existing Tasks
------------------
Existing tasks within Agent include a range of popular benchmarks, both single-step and multi-step:

- **ALFWorld**: A multi-step sequential decision-making householding task. The agent is provided text-based observations and must navigate a 3D household environment and interact with objects to complete a goal.
- **BabyAI**: A multi-step sequential decision-making task where the agent is provided text-based observations and must navigate a 2D grid world and interact with objects to complete a goal.
- **GSM8K**: A single-step question-answering task where the agent is provided grade school mathematical questions and must generate answers.
- **HotpotQA**: A single-step question-answering task where the agent is provided a question and a context and must generate an answer.
- **HumanEval**: A single-step question-answering task where the agent is provided a function definition and test cases and must generate the function definition.
- **WebShop**: A multi-step sequential decision-making web-based task. Given a text-based product requirement, the agent must navigate webpages and generate actions to find and purchase items.

Their task classes can be found under `src/agent/tasks/`:code: in the source code, while the task configurations can be found under `configs/tasks/`:code:.

Custom Tasks
----------------
In addition to the above existing tasks, you can easily define your own tasks, by subclassing the `Task`:code: class, defining a new task configuration, and adding the necessary prompt templates for the :doc:`methods`` you want to support.

Please see the `Creating a New Task <https://gitlab-uk.rnd.huawei.com/ai-uk-team/reinforcement_learning_london/agent/agent/-/tree/main/tutorials/create_new_task.ipynb/>`_ tutorial for a guided example on creating and using a new task.
