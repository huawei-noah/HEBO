import json
import time
import uuid
from typing import Any, Callable

import jellyfish

import agent.train.worker_distributed_hf as wdf
from agent.models.llm import LanguageBackend
from agent.train.replay_buffer import add_trajectory


class DistributedHFLanguageBackend(LanguageBackend):
    def __init__(
        self,
        model_id: str,
        logger: Any,
        client,
        server_only,
        **kwargs,
    ):
        # TODO: try to find context length
        context_length = 4096
        super().__init__(model_id, logger, context_length)

        self.client = client

        self.client_id = str(uuid.uuid4())

        self.generation_kwargs = kwargs

    def chat_completion(self, messages: list[dict[str, str]], parse_func: Callable, **kwargs) -> str:
        result = wdf.chat_completion(
            self.client,
            messages=messages,
            generation_kwargs={**self.generation_kwargs, **kwargs},
        )

        while not result.ready():
            time.sleep(0.2)

        raw_output_text = result.get()
        parsed_response = parse_func(raw_output_text)

        self.history.append({"input": messages, "output": raw_output_text, "parsed_response": parsed_response})

        self.logger.log_metrics(
            {
                "llm:input": messages,
                "llm:output": raw_output_text,
                "llm:parsed_response": parsed_response,
            }
        )

        return parsed_response

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

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        # TODO: untested because it's not being called?

        job_id = str(uuid.uuid4())  # Generate a unique job ID
        job_type = "count_tokens"

        job_queue = self.job_queue_prefix + self.model_id + ":" + job_type

        data = {
            "job_id": job_id,
            "model_id": self.model_id,
            "type": job_type,
            "args": {
                "messages": messages,
            },
        }
        job_data_str = json.dumps(data)

        self.client.rpush(job_queue, job_data_str)

        result_key = f"{self.result_key_prefix}{job_id}"
        response = json.loads(self.client.blpop(result_key)[1])
        self.client.delete(result_key)

        return response["data"]["tokens"]

    def save_trajectory_to_redis(self, trajectory):
        add_trajectory(self.client, trajectory=trajectory)
