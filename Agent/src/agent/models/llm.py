from abc import ABC
from abc import abstractmethod
from typing import Any, Callable


class LanguageBackend(ABC):
    def __init__(self, model_id: str, logger: Any, context_length: int) -> None:
        self.model_id = model_id
        self.logger = logger
        self.context_length = context_length
        self.history = []

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
    def chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> str:
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

    @abstractmethod
    def choose_from_options(
        self, messages: list[dict[str, str]], options: list[str], parse_func: Callable, **kwargs
    ) -> str:
        """Asks the model to choose from a list of options based on the given prompt.

        Args:
            messages (list[dict[str, str]]): The input text prompt to present along with options.
            options (List[str]): A list of options for the model to choose from.
            parse_func (Callable): A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the choice-making process,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The chosen option. Can return either the option's text directly or its index in the list.
        """
        pass
