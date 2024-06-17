from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import numpy as np

from agent.commands.core import Break
from agent.commands.core import Command
from agent.memory import MemKey
from agent.utils.utils import break_word_split
from agent.utils.utils import extract_action_as_json
from agent.utils.utils import extract_command_as_json
from agent.utils.utils import extract_metric_as_json
from agent.utils.utils import extract_submission_as_json
from agent.utils.utils import extract_summary_as_json

PARSE_FUNC_MAP = {
    "break_word_split": partial(break_word_split, "Command"),
    "extract_action_as_json": extract_action_as_json,
    "extract_command_as_json": extract_command_as_json,
    "extract_summary_as_json": extract_summary_as_json,
    "extract_submission_as_json": extract_submission_as_json,
    "extract_metric_as_json": extract_metric_as_json,
    # TODO: use JSONOutputParser instead and deal with parse_func_id
    "identity": lambda x: x,
}


class Flow(ABC):
    """Base class representing a flow in the system. A flow directs the execution order of commands
    or sub-flows.

    Attributes:
        name (str): The name of the flow.
        description (str): A brief description of the flow.
        prompt_template (None | str): path to jinja file containing the prompt template associated to the flow
    """

    def __init__(self, name: str, description: str, prompt_template: str | None = None):
        """Initializes the Flow object with a name and description.

        Args:
            name: The name of the flow.
            description: A brief description of the flow.
            prompt_template (None | str): path to jinja file containing the prompt template associated to the flow
        """
        self.name = name
        self.description = description
        self.prompt_template = prompt_template

    @abstractmethod
    def reset(self) -> None:
        """Resets the flow to its initial state.

        This method should be overridden by subclasses.
        """
        ...

    @abstractmethod
    def step(self, agent) -> "Command | Flow | None":
        """Executes a step in the flow and moves towards the next command or sub-flow.

        Args:
            agent: An agent that carries out the commands or sub-flows.

        This method should be overridden by subclasses.
        """
        ...


class SequentialFlow(Flow):
    """A Flow subclass that represents a sequential execution of commands or sub-flows.

    Attributes:
        sequence (list): A list of commands or sub-flows to be executed in order.
    """

    def __init__(self, sequence: list[Flow | Command], name="sequential_flow", description="A sequence of actions"):
        """Initializes the SequentialFlow object with a sequence of commands or sub-flows, a name,
        and a description.

        Args:
            sequence (list): A list of commands or sub-flows to be executed in order.
            name (str): The name of the flow. Defaults to "sequential_flow".
            description (str): A brief description of the flow. Defaults to "A sequence of actions".
        """
        super().__init__(name, description)
        self.sequence = sequence
        self._prev_state: int | None = None

    def reset(self) -> None:
        """Resets the SequentialFlow and all its commands or sub-flows to their initial state."""
        for cmd in self.sequence:
            if isinstance(cmd, Flow):
                cmd.reset()
        self._prev_state = None

    def step(self, agent) -> Command | Flow | None:
        """Finds the next command or sub-flow in the sequence and returns it.

        Args:
            agent: The agent which will execute the returned command.

        Returns:
            The next command to be executed or None if the end of the sequence is reached.
        """
        if self._prev_state is None:
            state = 0
        elif isinstance(self.sequence[self._prev_state], Flow):
            cmd = self.sequence[self._prev_state].step(agent)
            if cmd is not None:
                return cmd
            state = self._prev_state + 1  # sub-flow has completed
        else:
            state = self._prev_state + 1

        if state >= len(self.sequence):
            return None

        self._prev_state = state
        if isinstance(self.sequence[state], Command):
            agent.logger.log_metrics({f"flow:{repr(self)}:next_cmd": self.sequence[state].name})
            return self.sequence[state]

        return self.sequence[state].step(agent)


class DecisionFlow(Flow):
    """A Flow subclass representing a decision point where one command or sub-flow is chosen from
    multiple choices using the agent's LLM (Language Model).

    The LLM is prompted using a template (`flow_select.jinja`) to make a choice
    among the available commands or sub-flows. By default, the template makes use
    of the "name" and "description" attributes of sub-flows/commands.

    Attributes:
        choices (list): A list of commands or sub-flows from which one will be chosen and executed.
    """

    def __init__(
            self,
            choices: list[Flow | Command],
            name="decision_flow",
            description="Decide between choices",
            prompt_template="flow_select.jinja",
            max_retries: int = 1,
            parse_func_id: str = "break_word_split",
            max_tokens: Optional[int] = None,
            memory_choice_tag_val: Optional[str] = None,
            human_takeover: bool = True
    ):
        """Initializes the DecisionFlow object with multiple choices, a name, and a description.

        Args:
            choices (list): A list of commands or sub-flows from which one will be chosen and executed.
            name (str): The name of the flow. Defaults to "decision_flow".
            description (str): A brief description of the flow. Defaults to "Decide between choices".
            prompt_template (str): name of the jinja prompt template file for selection
            max_retries (int): number of retries allowed if choosing from options
            parse_func_id (str): id of the function to use to parse the agent output and extract its choice
            max_tokens (int): max number of tokens expected in the agent output - may be useful to get short answers
            memory_choice_tag_val (str): when taking a decision, check in the agent's memory to retrieve choices
            human_takeover (bool): whether human should provide the output if the LLM keeps getting parsing errors
        """
        super().__init__(name=name, description=description)

        self.choices = choices
        names = {c.name for c in self.choices}
        if len(self.choices) != len(names):
            count_choice = {name: 0 for name in names}
            for choice in self.choices:
                count_choice[choice.name] += 1
            duplicates = {name: value for name, value in count_choice.items() if value > 1}
            raise RuntimeError(f"All sub-flows in a decision point must have unique name. Got duplicates: {duplicates}")

        self._choice_taken = None
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.parse_function = PARSE_FUNC_MAP[parse_func_id]
        self.max_tokens = max_tokens
        if memory_choice_tag_val is not None:
            # check that it is a memory key
            assert memory_choice_tag_val in MemKey.list(), memory_choice_tag_val
            memory_choice_tag = MemKey.rev_dict()[memory_choice_tag_val]
        else:
            memory_choice_tag = None
        self.memory_choice_tag = memory_choice_tag
        self.human_takeover = human_takeover

    def reset(self) -> None:
        """Resets the DecisionFlow and all it sub-flows to their initial state."""
        for cmd in self.choices:
            if isinstance(cmd, Flow):
                cmd.reset()
        self._choice_taken = None

    def step(self, agent) -> Command | Flow | None:
        """Uses the agent's LLM to choose and executes one command or sub-flow from the available
        choices. The LLM is prompted using a `flow_select.jinja` template to make the decision.

        Args:
            agent: The agent which will execute the returned command. Its LLM is used for the decision.

        Returns:
            The chosen command (either directly the choice, or the command after following sub-flows)
            or None if the end of flow/sub-flows has been reached.
        """
        if isinstance(self._choice_taken, Break):
            return self._choice_taken
        if isinstance(self._choice_taken, Command):
            return None
        elif isinstance(self._choice_taken, Flow):
            return self._choice_taken.step(agent)

        choices = self.choices
        agent_choices = []
        if self.memory_choice_tag is not None:
            agent_choices = agent.memory.retrieve(tags={self.memory_choice_tag: 1.0})

        if len(agent_choices) > 0:
            choices = [c for c in choices if c.name in agent_choices]

        assert len(choices) > 0
        prompt_kwargs = {"memory": agent.memory, "flows": choices}
        if len(choices) == 1:
            selected_command = choices[0].name
            # need to update history
            output = f"@NOT GENERATED BY LLM AS IT WAS THE ONLY CHOICE@: {selected_command}"
            messages = agent.prompt_builder([self.prompt_template], prompt_kwargs)
            agent.llm.history.append({"input": messages, "output": output, "parsed_response": selected_command})
        else:
            selected_command = agent.safe_choose_from_options(
                ask_template=self.prompt_template,
                parse_func=self.parse_function,
                format_error_message='Answer using appropriate format.',
                options=[c.name for c in choices],
                prompt_kwargs=prompt_kwargs,
                human_takeover=self.human_takeover,
                max_retries=self.max_retries,
                max_tokens=self.max_tokens,
            )

        if os.getenv("DECISION_FLOW_BREAKPOINT"):
            messages = agent.prompt_builder([self.prompt_template], prompt_kwargs)
            print("==================================================")
            print("===================== PROMPT =====================")
            print("==================================================")
            for m in messages:
                print(f"[[ {m['role']} ]]")
                print(f"{m['content']}")
            print("==================================================")
            print("==================== RESPONSE ====================")
            print("==================================================")
            print(selected_command)
            breakpoint()

        self._choice_taken = choice = next((x for x in choices if x.name == selected_command), None)
        if choice is None:
            raise RuntimeError("Parsed output did not correspond to any of the available commands.")

        if isinstance(choice, Flow):
            return choice.step(agent)

        return choice


class LoopFlow(Flow):
    """A Flow subclass that represents a loop execution of commands or sub-flows that repeats
    itself at most max_repetitions times."""

    def __init__(
            self,
            loop_body: Flow | Command,
            max_repetitions: int,
            allow_early_break: bool,
            max_retries: int = 1,
            prompt_template: str = "flow_loop_continue.jinja",
            name: str = "loop_flow",
            description: str = "A loop that will execute a flow or a command for a number of steps",
            parse_func_id: str = "break_word_split",
            memory_choice_tag_val: Optional[str] = None,
            human_takeover_step: Optional[int] = None
    ):
        """Initializes the LoopFlow object with a sequence of commands or sub-flows, the max number
        of iterations, whether the loop can be terminated early by asking the LLMs, name, and a
        description.

        Args:
            loop_body: A flow or a command that will be repeated a number of times.
            max_repetitions: The maximum number of times that the sequence will be repeated.
            allow_early_break: If true, ask the LLM at the end of the sequence whether to break from the loop.
            max_retries: number of retries allowed to choose from options if LLM doesn't respond in correct format
            prompt_template: name of the jinja prompt template file for selection
            name: The name of the flow. Defaults to "sequential_flow".
            description: A brief description of the flow. Defaults to "A sequence of actions".
            parse_func_id: id of the function to use to extract the agent choice when a decision needs to be made
            memory_choice_tag_val: when taking a decision, check in the agent's memory to retrieve choices
            human_takeover_step: human answers after the k-th iteration of this loop (to prevent LLM repetitive failure)
        """
        super().__init__(name=name, description=description)
        self.loop_body = loop_body
        self.max_repetitions = max_repetitions
        self.allow_early_break = allow_early_break
        self.max_retries = max_retries
        self.prompt_template = prompt_template
        self.current_repeat = -1
        self.parse_function = PARSE_FUNC_MAP[parse_func_id]
        if self.allow_early_break:
            self.choices = ["Terminate", "Continue"]
        self.previous_command = None
        if memory_choice_tag_val is not None:
            # check that it is a memory key
            assert memory_choice_tag_val in MemKey.list(), memory_choice_tag_val
            memory_choice_tag = MemKey.rev_dict()[memory_choice_tag_val]
        else:
            memory_choice_tag = None
        self.memory_choice_tag = memory_choice_tag
        if human_takeover_step is None:
            human_takeover_step = np.inf
        self.human_takeover_step = human_takeover_step
        self.n_iter_without_human = 0

    def reset(self) -> None:
        """Resets the LoopFlow and all its commands or sub-flows to their initial state."""
        if isinstance(self.loop_body, Flow):
            self.loop_body.reset()
        self.current_repeat = -1
        self.previous_command = None

    def reset_internal(self) -> None:
        """Reset all the internal flows in the sequences."""
        if isinstance(self.loop_body, Flow):
            self.loop_body.reset()

    def step(self, agent) -> Command | Flow | None:
        """Either executes the flow inside the loop sequence or returns a command. If the end of
        sequence has been reached it checks whether the loop should be continued.

        Args:
            agent: The agent which will execute the returned command.

        Returns:
            The next command to be executed or None if the end of the sequence is reached.
        """
        if self.previous_command is None:
            self.current_repeat += 1
            self.n_iter_without_human += 1
            if self.current_repeat == self.max_repetitions:  # if max repetition is reached exit the loop
                return None

            from agent.agents import LLMAgent

            if self.n_iter_without_human == self.human_takeover_step and isinstance(agent, LLMAgent):
                agent.llm.human_mode = True
                self.n_iter_without_human = 0  # reset

            if (
                    self.allow_early_break and self.current_repeat > 0
            ):  # ask the llm whether to terminate the loop after having performed one step
                choices = self.choices
                agent_choices = []
                if self.memory_choice_tag is not None:
                    agent_choices = agent.memory.retrieve(tags={self.memory_choice_tag: 1.0})

                if len(agent_choices) > 0:
                    choices = [choice for choice in choices if choice in agent_choices]

                assert len(choices) > 0
                prompt_kwargs = {
                    "memory": agent.memory, "loop_body": self.loop_body, "current_repeat": self.current_repeat
                }

                if len(choices) == 1:
                    messages = agent.prompt_builder(
                        [self.prompt_template],
                        {"memory": agent.memory, "loop_body": self.loop_body, "current_repeat": self.current_repeat},
                    )
                    selected_choice = choices[0]
                    # need to update history
                    output = f"@NOT GENERATED BY LLM AS IT WAS THE ONLY CHOICE@: {selected_choice}"
                    agent.llm.history.append({"input": messages, "output": output, "parsed_response": selected_choice})
                else:
                    selected_choice = agent.safe_choose_from_options(
                        ask_template=self.prompt_template,
                        options=choices,
                        parse_func=self.parse_function,
                        format_error_message='Answer using appropriate format.',
                        prompt_kwargs=prompt_kwargs,
                        max_retries=self.max_retries,
                        human_takeover=self.human_takeover_step > 0,
                    )
                if selected_choice == "Continue":
                    self.reset_internal()
                else:
                    return None
            else:  # if early termination not allowed and max repetitions not reached, continue
                self.reset_internal()

        if isinstance(self.loop_body, Flow):
            self.previous_command = self.loop_body.step(agent)
            if self.previous_command is not None:
                return self.previous_command
            else:
                return self.step(agent)
        self.previous_command = None
        return self.loop_body


def run_flow(flow: Flow, agent):
    if flow is None:
        return
    flow.reset()
    while True:
        cmd = flow.step(agent)
        if cmd:
            cmd(agent)
        else:
            return
