from typing import Any, Dict

from datasets import load_dataset

from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import DatasetOutOfBoundsException
from agent.tasks import Task
from agent.utils import break_word_split
from agent.utils import normalize_answer
from agent.utils import pairwise_similarity


class HotpotQA(Task):
    def __init__(self, split: str, include_context: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.action_space = ActionSpace.CONTINUOUS
        self.args = kwargs
        self.dataset = load_dataset("hotpot_qa", "distractor", split=split)
        self.episode_counter = 0
        self.include_context = include_context

    def reset(self, next_subtask: str | None = None) -> dict[str, str]:
        """Reset the environment and return the initial observation."""

        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        if self.episode_counter > len(self.dataset):
            raise DatasetOutOfBoundsException(
                "The dataset index is not within dataset bounds. The end of the dataset may have been reached."
            )

        data = self.dataset[self.episode_counter]
        self.answer = data["answer"].strip()
        return self._return_observation(data)

    def answer_parser(self, raw_response: str):
        if "Answer: " in raw_response:
            return break_word_split("Answer", raw_response)
        return ""

    def step(self, action: str) -> tuple[dict, float, bool]:
        """Perform an action and return the next observation, reward, and done."""

        norm_answer = normalize_answer(self.answer)
        score = (
            0
            if norm_answer in ["yes", "no", "noanswer"] and normalize_answer(action) != norm_answer
            else pairwise_similarity(action, self.answer)
        )
        self.episode_counter += 1
        return {}, score, True

    def _return_observation(self, data: Dict[str, Any]) -> dict[str, str]:
        """Return the observation for the current step."""

        text_obs = []
        if self.include_context:
            text_obs = ["Here is the context you need for answering the question:", "Context:"]
            for title, sentences in zip(data["context"]["title"], data["context"]["sentences"]):
                text_obs.append("Title: " + title)
                text_obs.append("".join(sentences) + "\n")
            text_obs.append("\n")
        text_obs.append("Here is the question that you have to answer:")
        text_obs.append("Question: " + data["question"])
        text_obs = "\n".join(text_obs)

        return {MemKey.OBSERVATION: text_obs}
