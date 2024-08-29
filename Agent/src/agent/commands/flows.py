from functools import partial

from agent.commands.core import Command
from agent.utils import break_word_split


class Flow:
    """Base class representing a flow in the system. A flow directs the execution order of commands
    or sub-flows.

    Attributes:
        name (str): The name of the flow.
        description (str): A brief description of the flow.
    """

    def __init__(self, name, description):
        """Initializes the Flow object with a name and description.

        Args:
            name (str): The name of the flow.
            description (str): A brief description of the flow.
        """
        self.name = name
        self.description = description

    def reset(self):
        """Resets the flow to its initial state.

        This method should be overridden by subclasses.
        """
        ...

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

    def reset(self):
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

    def __init__(self, choices: list[Flow | Command], name="decision_flow", description="Decide between choices"):
        """Initializes the DecisionFlow object with multiple choices, a name, and a description.

        Args:
            choices (list): A list of commands or sub-flows from which one will be chosen and executed.
            name (str): The name of the flow. Defaults to "decision_flow".
            description (str): A brief description of the flow. Defaults to "Decide between choices".
        """
        super().__init__(name, description)

        self.choices = choices
        assert len(self.choices) == len(
            {c.name for c in self.choices}
        ), "All sub-flows in a decision point must have unique name."

        self._choice_taken = None

    def reset(self):
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
        if isinstance(self._choice_taken, Command):
            return None
        elif isinstance(self._choice_taken, Flow):
            return self._choice_taken.step(agent)

        selected_command = agent.llm.choose_from_options(
            messages=agent.prompt_builder(["flow_select.jinja"], {"memory": agent.memory, "flows": self.choices}),
            options=[c.name for c in self.choices],
            parse_func=partial(break_word_split, "Command"),
        )
        self._choice_taken = choice = next((x for x in self.choices if x.name == selected_command), None)
        assert choice, "Parsed output did not correspond to any of the available commands."

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
        name: str = "loop_flow",
        description: str = "A loop that will execute a flow or a command for a number of steps",
    ):
        """Initializes the LoopFlow object with a sequence of commands or sub-flows, the max number
        of iterations, whether the loop can by terminated early by asking the llms, name, and a
        description.

        Args:
            loop_body: A flow or a command that will be repeated a number of times.
            max_repetitions: The maximum number of times that the sequence will be repeated.
            allow_early_break: If true, ask the LLM at the end of the sequence whether to break from the loop.
            name: The name of the flow. Defaults to "sequential_flow".
            description: A brief description of the flow. Defaults to "A sequence of actions".
        """
        super().__init__(name, description)
        self.loop_body = loop_body
        self.max_repetitions = max_repetitions
        self.allow_early_break = allow_early_break
        self.current_repeat = -1
        if self.allow_early_break:
            self.choices = ["Terminate", "Continue"]

    def reset(self):
        """Resets the LoopFlow and all its commands or sub-flows to their initial state."""
        if isinstance(self.loop_body, Flow):
            self.loop_body.reset()
        self.current_repeat = -1
        self.previous_command = None

    def reset_internal(self) -> None:
        """Reset all the internal flows in the sequences."""
        if isinstance(self.loop_body, Flow):
            self.loop_body.reset()

    def step(self, agent) -> Command | None:
        """Either executes the flow inside the loop sequence or returns a command. If the end of
        sequence has been reached it checks whether the loop should be continued.

        Args:
            agent: The agent which will execute the returned command.

        Returns:
            The next command to be executed or None if the end of the sequence is reached.
        """
        if self.previous_command is None:
            self.current_repeat += 1
            if self.current_repeat == self.max_repetitions:  # if max repetition is reached exit the loop
                return None
            elif (
                self.allow_early_break and self.current_repeat > 0
            ):  # ask the llm whether to terminate the loop after having performed one step
                selected_command = agent.llm.choose_from_options(
                    messages=agent.prompt_builder(
                        ["flow_loop_continue.jinja"],
                        {"memory": agent.memory, "loop_body": self.loop_body, "current_repeat": self.current_repeat},
                    ),
                    options=self.choices,
                    parse_func=partial(break_word_split, "Answer"),
                )
                if selected_command == "Continue":
                    self.reset_internal()
                else:
                    return None
            else:  # if early termination is not allowed and the max repetitions have not be reach continue in the loop
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
