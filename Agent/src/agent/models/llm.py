import dataclasses
import os
import traceback
from abc import ABC
from abc import abstractmethod
from typing import Any, Callable

from agent.utils.utils import human_input


class LanguageBackend(ABC):
    def __init__(self, model_id: str, logger: Any, context_length: int) -> None:
        self.model_id = model_id
        self.logger = logger
        self.context_length = context_length
        self.history = []
        self._human_mode = False

    @property
    def human_mode(self) -> bool:
        return self._human_mode

    @abstractmethod
    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Counts the number of tokens in a given text according to the model's tokenizer.

        Args:
            messages (list[dict[str, str]]): The list of messages to count tokens for.

        Returns:
            int: The number of tokens after encoding the prompt.
        """
        pass

    @abstractmethod
    def _chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> str:
        """Generates a text completion for a given prompt in a chat-like interaction.

        Args:
            messages (list[dict[str, str]]): The input text prompt to generate completion for.
            parse_func (Callable): A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the generation,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The generated text completion.
        """
        pass

    @staticmethod
    def message_to_human_str(message: dict[str, Any]) -> str:
        """ Convert a message object into a readable string """
        role = message["role"]
        content = message["content"]
        return f"[[{role.upper()}]]\n{content}"

    def human_chat_completion(self, messages: list[dict[str, str]], parse_func: Callable) -> str:
        """Fakes an LLM generation by getting input from the keyboard.

        Args:
            messages (list[dict[str, str]]): The input text prompt to generate completion for.
            parse_func (Callable): A function to parse the model's response.

        Returns:
            str: The text completion.
        """
        if os.getenv("NO_HUMAN", False):
            traceback.print_stack()
            print("Queried human input in NO_HUMAN mode!")
            exit(1)
        stop_human_mode = "STOP MANUAL ASSISTANCE"

        for m in messages:
            for k, v in m.items():
                print(k, ":", v)

        print(
            f"\n{'-' * 50}\nHuman Assistance Mode.\n\n"
            f"To stop human assistance mode: add '{stop_human_mode}' anywhere in your reply\n{'-' * 50}"
        )

        reply = human_input(allow_cancel=False)
        if stop_human_mode in reply:
            self._human_mode = False  # stop manual assistance

        reply = reply.replace(stop_human_mode, "")

        parsed_response = parse_func(reply)
        return parsed_response


class HumanSupervisionAction:
    """ Type of action that a user can choose when user supervision is queried """
    keys: list[str]
    description: str


class HumanGuidanceAction(HumanSupervisionAction):
    """ Type of action that a user can choose when user supervision is queried """
    keys = ["g", "G"]
    description = ("Add indications at the end of the prompt and let the LLM generate "
                   "a new answer (previous indications will be overwritten).")


class HumanContinueAction(HumanSupervisionAction):
    keys = ["c", "C"]
    description = "Continue without changing LLM output."


class HumanResampleLLMAction(HumanSupervisionAction):
    keys = ["r", "R"]
    description = "Make the LLM generate a new response."


class HumanFixAction(HumanSupervisionAction):
    keys = ["f", "F"]
    description = "Fix the LLM raw output (this will trigger human editing mode)."


class HumanSeqPrintModeAction(HumanSupervisionAction):
    keys = ["h", "H"]
    description = "Print messages one after the other (easier to copy paste or for small window width)"


@dataclasses.dataclass
class HumanSupervisionChecker:
    allow_resampling: bool
    allow_guidance: bool
    allow_continue: bool
    allow_fix: bool
    allow_seq_print: bool
    length: int = 150

    def query_human(self) -> HumanSupervisionAction:
        """
            Let human interact with the system (user can decide to continue, fix LLM output, redo LLM generation, etc.)

            Returns:
                action: the action chosen by the user
            """
        if os.getenv("NO_HUMAN", False):
            traceback.print_stack()
            print("Queried human input in NO_HUMAN mode!")
            exit(1)

        lines = ["┌" + "─" * (self.length - 2) + "┐"]
        new_line = "│ [HUMAN SUPERVISION] Choose an option:"
        new_line += " " * (self.length - 1 - len(new_line)) + "│"
        lines.append(new_line)
        actions = self.get_actions()
        options = {}
        for action in actions:
            new_line = f"│ - {'/'.join(action.keys)}: {action.description}"
            new_line += " " * (self.length - 1 - len(new_line)) + "│"
            lines.append(new_line)
            for new_key in action.keys:
                assert new_key not in options, f"duplicate key {new_key} for {options[new_key]} and {action}"
                options[new_key] = action
        lines.append("└" + "─" * (self.length - 2) + "┘")
        print("\n".join(lines))
        control = ""
        while control not in options:
            control = input(f"Write the option ({'/'.join(options)}):").strip()
        return options[control]

    def get_actions(self) -> list[HumanSupervisionAction]:
        actions: list[HumanSupervisionAction] = []
        if self.allow_continue:
            actions.append(HumanContinueAction())
        if self.allow_fix:
            actions.append(HumanFixAction())
        if self.allow_resampling:
            actions.append(HumanResampleLLMAction())
        if self.allow_guidance:
            actions.append(HumanGuidanceAction())
        if self.allow_seq_print:
            actions.append(HumanSeqPrintModeAction())
        return actions
