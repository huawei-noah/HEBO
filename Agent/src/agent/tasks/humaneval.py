from typing import Any, Dict

from human_eval.data import read_problems
from human_eval.execution import check_correctness

from agent.memory import MemKey
from agent.tasks import ActionSpace
from agent.tasks import DatasetOutOfBoundsException
from agent.tasks import Task
from agent.utils import pylogger

log = pylogger.get_pylogger(__name__)


class HumanEval(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_space = ActionSpace.CONTINUOUS
        self.problems = list(read_problems().items())[3:]  # first 3 used for FS
        self.episode_counter = 0

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        """Reset the environment and return the initial observation."""

        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        if self.episode_counter > len(self.problems):
            raise DatasetOutOfBoundsException(
                "The dataset index is not within dataset bounds. The end of the dataset may have been reached."
            )

        self.problem = self.problems[self.episode_counter][1]
        self.episode_counter += 1
        return self._return_observation(self.problem)

    def _return_observation(self, problem: Dict[str, Any]) -> Dict[str, str]:
        """Return the observation for the current step."""

        return {MemKey.OBSERVATION: problem["prompt"]}

    def answer_parser(self, raw_response):
        try:
            start = raw_response.index("[BEGIN]") + len("[BEGIN]")
            end = raw_response.index("[END]")
            proposed_answer = raw_response[start:end]
        except ValueError:
            proposed_answer = raw_response
        return proposed_answer

    def step(self, action: str) -> tuple[dict, float, bool]:
        result = check_correctness(self.problem, action, timeout=3.0)
        score = 1 if result["passed"] else 0
        return {}, score, True
