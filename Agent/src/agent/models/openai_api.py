import hashlib
import os
import time
from typing import Any, Callable

import numpy as np
import openai
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChatCompletion
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_fixed

from agent.models.embeddings import EmbeddingBackend
from agent.models.llm import LanguageBackend, HumanFixAction, HumanContinueAction, \
    HumanResampleLLMAction, HumanGuidanceAction, HumanSupervisionChecker, HumanSeqPrintModeAction
from agent.utils.utils import human_input, print_in_two_cols, HumanInputCancelError, save_w_pickle, load_w_pickle


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
    def __init__(self, model_id: str, logger: Any, context_length: int, api_key: str, server_ip: str,
                 responses_from_file: list[str] | None = None, **kwargs):
        """
        Args:
            responses_from_file:  if set, the answers will be taken from the list rather than queried from the LLM
        """
        super().__init__(model_id=model_id, logger=logger, context_length=context_length)
        self.api_key = api_key
        self.base_url = server_ip
        self.generation_kwargs = kwargs
        self.responses_from_file = responses_from_file
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.human_guidance = ""  # save the potential guidance added during human supervision

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.RateLimitError)),
    )
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
        path_to_saved_responses: str | None = kwargs.pop("path_to_saved_responses", None)

        try:
            if not self.human_mode:
                try:
                    # Before querying the LLM, check if the response was not given previously
                    name = self.model_id + "_" + hashlib.md5(str(messages).encode('utf-8')).digest().hex() + ".pkl"
                    save_response_filepath = None
                    if path_to_saved_responses is not None:
                        save_response_filepath = os.path.join(path_to_saved_responses, name)
                    response = None
                    if save_response_filepath is not None and os.path.exists(save_response_filepath):
                        try:
                            response = load_w_pickle(path=save_response_filepath)
                            self.logger.log_metrics({"llm:retrieved_response": True})
                        except:
                            pass

                    if response is None:
                        if self.responses_from_file is not None and len(self.responses_from_file) > 0:
                            resp_message = ChatCompletionMessage(content=self.responses_from_file.pop(0),
                                                                 role="assistant")
                            choices = [Choice(finish_reason="stop", index=0, message=resp_message)]
                            response = ChatCompletion(
                                id="READ_FROM_FILE",
                                choices=choices, object="chat.completion",
                                created=time.time(),
                                model="READ_FROM_FILE",
                                usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
                            )
                        else:
                            # Query the LLM
                            response = self.client.chat.completions.create(
                                model=self.model_id,
                                messages=messages,
                                **self.generation_kwargs,
                                **kwargs,
                            )
                            if save_response_filepath is not None:
                                # couldn't use json with ChatCompletion type...
                                save_w_pickle(obj=response, path=save_response_filepath, filename=None, overwrite=False)

                except openai.BadRequestError as e:
                    # check if the message is too long
                    import re
                    max_len_err = re.findall("This model\'s maximum context length is [0-9]+ tokens", e.message)
                    if len(max_len_err) > 0:
                        # max_length = int(max_len_err[0].split()[-2])
                        # TODO: finish this: shrink the message or handle that in another way
                        raise
                    else:
                        raise
                raw_output_text = response.choices[0].message.content.strip()
                api_usage_input = response.usage.prompt_tokens
                api_usage_output = response.usage.completion_tokens
            else:
                raw_output_text = super().human_chat_completion(
                    messages=messages, parse_func=lambda x: x
                )
                api_usage_input = 0
                api_usage_output = 0
        except openai.APIConnectionError as e:
            print(f"API Connection Error: {e}")
            raise
        except openai.RateLimitError as e:
            print(f"Rate Limit Exceeded: {e}")
            raise
        except openai.OpenAIError as e:
            print(f"General OpenAI API Error: {e}")
            raise
        human_mode = self.human_mode

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": raw_output_text,
                "api_usage:input": api_usage_input,
                "api_usage:output": api_usage_output,
            }
        )

        human_supervise = os.getenv("HUMAN_SUPERVISE", False)
        if human_supervise in ["0", "False"]:
            human_supervise = False
        screen_width = 186
        prompt_to_display = ""
        raw_response_to_display = ""

        if human_supervise and not self.human_mode:
            prompt_to_display += "================================================\n"
            prompt_to_display += "==================== PROMPT ====================\n"
            prompt_to_display += "================================================\n"
            for m in messages:
                prompt_to_display += f"[[ {m['role']} ]]\n"
                prompt_to_display += f"{m['content']}\n"

            raw_response_to_display += "====================================================\n"
            raw_response_to_display += "=================== RAW RESPONSE ===================\n"
            raw_response_to_display += "====================================================\n"
            raw_response_to_display += raw_output_text + "\n"
        try:
            parsed_response = parse_func(raw_output_text)

        except Exception as e:
            if human_supervise and not self.human_mode:
                print_in_two_cols(t1=prompt_to_display, t2=raw_response_to_display, w=screen_width)
            self.logger.log_metrics(
                {
                    "llm:human_mode": human_mode,
                    "llm:parsed_error": e.args[0],
                }
            )
            self.logger.save_metrics()
            raise

        if human_supervise and not self.human_mode:
            response_to_display = ""
            if raw_output_text != parsed_response:
                response_to_display += raw_response_to_display + "\n"

            response_to_display += "=======================================================\n"
            response_to_display += "=================== PARSED RESPONSE ===================\n"
            response_to_display += "=======================================================\n"
            response_to_display += str(parsed_response)
            print_in_two_cols(t1=prompt_to_display, t2=response_to_display, w=screen_width)

            human_supervision_handler = HumanSupervisionChecker(
                allow_fix=True, allow_continue=True, allow_resampling=True, allow_guidance=True, allow_seq_print=True,
                length=screen_width - 1
            )

            while True:
                action = human_supervision_handler.query_human()

                if isinstance(action, HumanSeqPrintModeAction):
                    human_supervision_handler.allow_seq_print = False
                    print(prompt_to_display)
                    print(response_to_display)
                    action = human_supervision_handler.query_human()

                if isinstance(action, HumanContinueAction):
                    print("Continuing!\n")
                    break
                elif isinstance(action, HumanFixAction):
                    print("Get human fix!")
                    try:
                        fixed_reply = human_input(allow_cancel=True, w=screen_width)
                    except HumanInputCancelError:
                        print("Cancel human fix!")
                        continue
                    raw_output_text = fixed_reply
                    parsed_response = parse_func(raw_output_text)
                    human_mode = True

                    break
                elif isinstance(action, HumanResampleLLMAction):
                    print("Get a new answer!")
                    return self._chat_completion(messages=messages, parse_func=parse_func,
                                                 path_to_saved_responses=path_to_saved_responses, **kwargs)
                elif isinstance(action, HumanGuidanceAction):
                    print("Provide guidance to get a better answer!")
                    assert messages[-1]["role"] == "user", messages[-1]
                    try:
                        guidance = "\n" + human_input(allow_cancel=True, w=screen_width)
                    except HumanInputCancelError:
                        print("Cancel human guidance!")
                        continue
                    last_content = messages[-1]["content"]
                    if len(self.human_guidance) > 0 and last_content[-len(self.human_guidance):] == self.human_guidance:
                        last_content = messages[-1]["content"][:-len(self.human_guidance)]
                    self.human_guidance = guidance
                    last_content += guidance
                    messages[-1]["content"] = last_content
                    return self._chat_completion(
                        messages=messages, parse_func=parse_func, path_to_saved_responses=path_to_saved_responses,
                        **kwargs
                    )
                else:
                    raise ValueError(action)

        self.history.append({"input": messages, "output": raw_output_text, "parsed_response": parsed_response})
        self.logger.log_metrics(
            {
                "llm:parsed_response": parsed_response,
                "llm:human_mode": human_mode,
            }
        )
        self.logger.save_metrics()

        return parsed_response

    def count_tokens(self, prompt: list[dict[str, str]]) -> int:
        return _num_tokens_from_messages(prompt, self.model_id)


class OpenAIAPIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend using OpenAI's API."""

    def __init__(self, api_key: str | None, server_ip: str, **kwargs):
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
