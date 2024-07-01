import numpy as np

from agent.commands.core import Command
from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.utils import pairwise_similarity


class ExecutePlannedAction(Command):
    """Executes a previously planned action by retrieving it from memory and then acting on the
    environment.

    It clears the planned action from memory after execution.
    """

    name: str = "act"
    description: str = "Act on the environment"

    input_keys: dict[str, MemKey] = {"planned_action_mem_key": MemKey.NEXT_PLANNED_ACTION}
    renamed_keys: dict[MemKey, str] = {MemKey.NEXT_PLANNED_ACTION: MemKey.EXTERNAL_ACTION}
    deleted_keys: list[MemKey] = [MemKey.NEXT_PLANNED_ACTION, MemKey.NEXT_PLANNED_ACTION_DIVERSE]

    def func(self, agent, planned_action_mem_key: MemKey):
        planned_action = agent.memory.retrieve({planned_action_mem_key: 1.0})
        agent.external_action = planned_action


class ConsiderAction(Command):
    """Considers an action to be sent to the environment by generating a prompt with a specified
    template. The considered action is then stored in memory under a specific key, and this can be
    appended to a list of actions if required.

    Input:
        ask_template: "external_action.jinja" - The template used to generate the prompt for considering an action.

        append: bool - Determines if the action should be appended to a list of previously considered actions.

    Output:
        memory_key: MemKey.NEXT_PLANNED_ACTION - The key under which the considered action is stored.
    """

    # COMMAND DEFINITION
    name: str = "consider_act"
    description: str = "Consider the action that will be sent to the environment."

    input_keys: dict[str, MemKey] = {"avail_actions_mem_key": MemKey.AVAILABLE_ACTIONS}
    output_keys: dict[str, MemKey] = {"output_mem_key": MemKey.NEXT_PLANNED_ACTION}

    required_prompt_templates: dict[str, str] = {"ask_template": "external_action.jinja"}

    # ConsiderAction Specifics:
    append: bool = False

    def func(self, agent, avail_actions_mem_key, output_mem_key, ask_template):
        if agent.task.action_space == ActionSpace.DISCRETE:
            assert agent.memory.retrieve({avail_actions_mem_key: 1.0}) is not None
            output = agent.llm.choose_from_options(
                agent.prompt_builder([ask_template], {"memory": agent.memory}),
                agent.memory.retrieve({avail_actions_mem_key: 1.0}),
                agent.task.answer_parser,
            )
        else:
            output = agent.llm.chat_completion(
                agent.prompt_builder([ask_template], {"memory": agent.memory}), agent.task.answer_parser
            )

        if self.append:
            prev_actions = agent.memory.retrieve({output_mem_key: 1.0})
            assert prev_actions is None or isinstance(prev_actions, list)

            new_actions = [output] + prev_actions if prev_actions else [output]
            agent.memory.store(new_actions, {output_mem_key})
        else:
            agent.memory.store(output, {output_mem_key})


class ReflectOnPlannedAction(ConsiderAction):
    """Reflects on a planned action by asking the model to confirm its previous answer, using a
    specific template. The reflection or confirmation is then stored under the same memory key as
    the initial planned action.

    Input:
        ask_template: "zerostep_reflect.jinja" - The template used to generate the prompt for reflection.

        memory_key: MemKey.NEXT_PLANNED_ACTION

    Output:
        memory_key: MemKey.NEXT_PLANNED_ACTION - Stores the reflection output as a new planned action.
    """

    name: str = "zero_step_reflect"
    description: str = "Ask the model if its sure of its answer and confirm it."

    required_prompt_templates: dict[str, str] = {"ask_template": "zerostep_reflect.jinja"}


class ConsistencyOnDiverseActions(Command):
    """Finds the most consistent action among a set of considered actions. The selection process
    varies depending on the action space (discrete or continuous), including similarity
    calculations for non-discrete actions.

    Input:
        memory_key: MemKey.NEXT_PLANNED_ACTION_DIVERSE - The list of considered actions.
    Output:
        memory_key: MemKey.NEXT_PLANNED_ACTION - Stores the most consistent action selected from the input.

        Clears memory_key: MemKey.NEXT_PLANNED_ACTION_DIVERSE after selection.
    """

    name: str = "self_consistency_act"
    description: str = "Find the most consistent of the considered actions."

    input_keys: dict[str, MemKey] = {"diverse_actions_mem_key": MemKey.NEXT_PLANNED_ACTION_DIVERSE}

    output_keys: dict[str, MemKey] = {"next_action_mem_key": MemKey.NEXT_PLANNED_ACTION}
    deleted_keys: list[MemKey] = [MemKey.NEXT_PLANNED_ACTION_DIVERSE]

    def func(self, agent, diverse_actions_mem_key, next_action_mem_key):
        diverse_actions = agent.memory.retrieve({diverse_actions_mem_key: 1.0})

        if agent.task.action_space == ActionSpace.DISCRETE:
            selected_action = max(set(diverse_actions), key=diverse_actions.count)

        else:
            parsed_diverse_actions = [action for action in diverse_actions if action]

            # calculate pairwise similarity among all responses
            all_similarity = np.array(
                [[pairwise_similarity(i, j) for i in parsed_diverse_actions for j in parsed_diverse_actions]]
            ).reshape(len(parsed_diverse_actions), len(parsed_diverse_actions))
            selected_action = (
                parsed_diverse_actions[all_similarity.sum(axis=1).argmax()] if parsed_diverse_actions else ""
            )  # same for axis=0, since all_similarity is a symmetric matrix

        agent.memory.store(selected_action, {next_action_mem_key})
