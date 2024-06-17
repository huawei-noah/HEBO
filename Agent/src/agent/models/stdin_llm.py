from typing import Callable

import jellyfish
from rich import print

from agent.models.llm import LanguageBackend
from agent.parsers.parser import ParsingError


class StdinLanguageBackend(LanguageBackend):

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return 0

    def chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> str:
        """Fakes an LLM generation by getting input from the keyboard.

        Args:
            messages (list[dict[str, str]]): The input text prompt to generate completion for.
            parse_func (Callable): A function to parse the model's response.
            **kwargs: Additional keyword arguments that may be required for the generation,
                      such as temperature, max_tokens, etc.

        Returns:
            str: The text completion.
        """

        if not self.human_mode:
            print(f"\n{'-' * 20}\nHere is a list of messages that would normally be fed to the LLM.\n{'-' * 20}")
            for m in messages:
                for k, v in m.items():
                    print(k, ":", v)

            reply = ""
            output = input("Please enter a reply: ")
            n_empty_lines_in_a_raw = 0
            stop_signal = "[STOP]"
            while stop_signal not in output[-len(stop_signal):]:
                reply += output + "\n"
                if output == "":
                    n_empty_lines_in_a_raw += 1
                else:
                    n_empty_lines_in_a_raw = 0
                if n_empty_lines_in_a_raw >= 5:
                    msg = f"print what's next (just print {stop_signal} to end):\n"
                else:
                    msg = ""
                output = input(msg)
            output = output[:len(output) - len(stop_signal)]
            reply += output
        else:
            reply = super().human_chat_completion(messages=messages, parse_func=lambda x: x)

        parsed_response = parse_func(reply)

        self.history.append({"input": messages, "output": reply, "parsed_response": parsed_response})

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": reply,
                "llm:parsed_response": parsed_response,
            }
        )

        return parsed_response

    def choose_from_options(
            self, messages: list[dict[str, str]], options: list[str], parse_func: Callable, **kwargs
    ) -> str:
        response = self.chat_completion(messages, parse_func, **kwargs)
        selected_option = min(options, key=lambda x: jellyfish.levenshtein_distance(response, x))
        self.history[-1]["parsed_response"] = selected_option
        self.logger.log_metrics(
            {
                "llm:input_options": options,
                "llm:chosen_option": selected_option,
            }
        )
        return selected_option
