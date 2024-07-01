from agent.commands.core import Command
from agent.memory import MemKey


class _SimpleThought(Command):
    """A base class for commands that generate a thought, reflection, or decomposition of a problem
    by utilizing a specified template to prompt a response and storing it in memory under a given
    key.

    Input:
        ask_template (str): A template filename used to generate the prompt.
                            This is a class variable that needs to be defined in subclasses.
                            Example values: "think.jinja", "reflect.jinja", "decompose.jinja".

    Output:
        memory_key (MemKey): The key under which the generated response is stored in the agent's memory.
                             This is a class variable that needs to be defined in subclasses.
                             Example values: MemKey.THOUGHT, MemKey.REFLECTION, MemKey.SUBPROBLEM.
    """

    # requires these two variables:
    output_keys: dict[str, MemKey] = {}
    required_prompt_templates: dict[str, str] = {}

    def func(self, agent, ask_template, output_mem_key):
        llm_input = agent.prompt_builder([ask_template], {"memory": agent.memory})
        response = agent.llm.chat_completion(llm_input, lambda x: x)
        agent.memory.store(response, {output_mem_key})


class Think(_SimpleThought):
    """Represents a command to consider the problem and create a plan, using a specific template
    for prompting and storing the result under a specific memory key.

    Input:
        ask_template: "think.jinja" - Specifies the template used to generate the prompt for thinking.

    Output:
        memory_key: MemKey.THOUGHT - The key under which the thought process's outcome is stored in memory.
    """

    name: str = "think"
    description: str = "Spend some time to consider the problem and create a plan."

    output_keys: dict[str, MemKey] = {"output_mem_key": MemKey.THOUGHT}
    required_prompt_templates: dict[str, str] = {"ask_template": "think.jinja"}


class Reflect(_SimpleThought):
    """Represents a command to reflect on past mistakes and formulate a new plan of action, using a
    specific template for prompting and storing the result under a specific memory key.

    Input:
        ask_template: "reflect.jinja" - Specifies the template used to generate the prompt for reflection.

    Output:
        memory_key: MemKey.REFLECTION - The key under which the reflection's outcome is stored in memory.
    """

    name: str = "reflect"
    description: str = "Reflect on past mistakes for a new plan of action."

    output_keys: dict[str, MemKey] = {"output_mem_key": MemKey.REFLECTION}
    required_prompt_templates: dict[str, str] = {"ask_template": "reflect.jinja"}


class Decompose(_SimpleThought):
    """Represents a command to decompose the problem into subproblems, using a specific template
    for prompting and storing the result under a specific memory key.

    Input:
        ask_template: "decompose.jinja" - Specifies the template used to generate the prompt for decomposition.

    Output:
        memory_key: MemKey.SUBPROBLEM - The key under which the decomposition's outcome is stored in memory.
    """

    name: str = "decompose"
    description: str = "Spend some time to decompose the problem to subproblems."

    output_keys: dict[str, MemKey] = {"output_mem_key": MemKey.SUBPROBLEM}
    required_prompt_templates: dict[str, str] = {"ask_template": "decompose.jinja"}
