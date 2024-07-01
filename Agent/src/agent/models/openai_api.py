from typing import Any, Callable

import jellyfish
import numpy as np
import openai
from openai import OpenAI
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_fixed

from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import LanguageBackend


def _num_tokens_from_messages(messages: list[dict[str, str]], model: str) -> int:
    """Return the number of tokens used by a list of messages.

    Args:
        messages (list[dict[str, str]]): The list of messages to count tokens for.

    Returns:
        int: The number of tokens after encoding the prompt.

    Taken from OpenAI's Cookbook: https://github.com/openai/openai-
    cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return _num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for
            information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class OpenAIAPILanguageBackend(LanguageBackend):
    def __init__(self, model_id: str, logger: Any, context_length: int, api_key: str, server_ip: str, **kwargs):
        super().__init__(model_id, logger, context_length)

        self.api_key = api_key
        self.base_url = server_ip
        self.generation_kwargs = kwargs

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(self.logger)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError)),
    )
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
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **self.generation_kwargs,
                **kwargs,
            )
        except openai.APIConnectionError as e:
            print(f"API Connection Error: {e}")
            raise
        except openai.RateLimitError as e:
            print(f"Rate Limit Exceeded: {e}")
            raise
        except openai.OpenAIError as e:
            print(f"General OpenAI API Error: {e}")
            raise

        raw_output_text = response.choices[0].message.content.strip()
        parsed_response = parse_func(raw_output_text)

        self.history.append({"input": messages, "output": raw_output_text, "parsed_response": parsed_response})

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": raw_output_text,
                "llm:parsed_response": parsed_response,
                "api_usage:input": response.usage.prompt_tokens,
                "api_usage:output": response.usage.completion_tokens,
            }
        )

        return parsed_response

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

    def count_tokens(self, prompt: list[dict[str, str]]) -> int:
        return _num_tokens_from_messages(prompt, self.model_id)


class OpenAIAPIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using OpenAI's API."""

    def __init__(self, api_key, server_ip, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = server_ip
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_text(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        ret_val = self.client.embeddings.create(input=[text], model=self.model_id)
        # TODO: log usage

        return np.array(ret_val.data[0].embedding)

    def batch_embed_texts(self, texts: list[str]) -> list[list[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        ret_val = self.client.embeddings.create(input=texts, model=self.model_id)
        # TODO: log usage
        # TODO fix following line
        return np.vstack(ret_val.data[0].embedding)
