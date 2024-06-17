import math
import re
from typing import Any, Dict

from datasets import load_dataset

from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import DatasetOutOfBoundsException
from agent.tasks import Task


class GSM8K(Task):
    def __init__(self, split: str, **kwargs):
        super().__init__(**kwargs)

        self.action_space = ActionSpace.CONTINUOUS
        self.args = kwargs
        self.dataset = load_dataset("gsm8k", "main", split=split)
        self.episode_counter = 0

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        """Reset the environment and return the initial observation."""

        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        if self.episode_counter > len(self.dataset):
            raise DatasetOutOfBoundsException(
                "The dataset index is not within dataset bounds. The end of the dataset may have been reached."
            )

        data = self.dataset[self.episode_counter]
        self.answer = float(data["answer"].split("\n####")[-1].replace(",", ""))
        return self._return_observation(data)

    def answer_parser(self, raw_response: str):
        try:
            proposed_answer = re.findall(r"[-+]?(?:\d*\.*\d+)", raw_response.replace(",", ""))[-1]
        except IndexError:
            proposed_answer = ""
        return proposed_answer

    def step(self, action: str) -> tuple[dict, float, bool]:
        """Perform an action and return the next observation, reward, and done."""

        try:
            reward = 1 if math.isclose(float(action), self.answer) else 0
        except Exception:
            reward = 0
        self.episode_counter += 1
        return {}, reward, True

    def _return_observation(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Return the observation for the current step."""

        return {MemKey.OBSERVATION: data["question"]}
