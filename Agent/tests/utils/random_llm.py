import random
from typing import Callable

from agent.models.llm import LanguageBackend


class RandomLanguageBackend(LanguageBackend):
    def __init__(self, **kwargs):
        super().__init__("random", None, 0)

    def chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> str:
        return "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do...."

    def choose_from_options(
        self, messages: list[dict[str, str]], options: list[str], parse_func: Callable, **kwargs
    ) -> str:
        """Asks the model to choose from a list of options based on the given prompt.

        Args:
            prompt (str): The input text prompt to present along with options.
            options (List[str]): A list of options for the model to choose from.
            **kwargs: Additional keyword arguments that may be required for the choice-making process,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The chosen option. Can return either the option's text directly or its index in the list.
        """
        return random.choice(options)

    def count_tokens(self, prompt: list[dict[str, str]]) -> int:
        return len(prompt)
