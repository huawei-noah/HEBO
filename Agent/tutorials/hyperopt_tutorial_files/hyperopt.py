import os
from typing import Dict, List

from agent.tasks.tasks import Task

from agent.memory import MemKey


class HyperOpt(Task):
    def __init__(self, workspace_path: str, **kwargs):
        super().__init__(**kwargs)
        self.episode_counter = 0
        self.test_scores: List[float] = []
        self.workspace_path = workspace_path
        self.done = False
        self.step_num = 0
        self.max_steps = 1
        self.id = 'hyperopt'

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        # self.workspace_path = input("input workspace directory:")
        assert os.path.exists(self.workspace_path)
        assert os.path.exists(self.workspace_data_path)
        assert os.path.exists(f"{self.workspace_path}/code")
        os.makedirs(f"{self.workspace_path}/results", exist_ok=True)

        self.done = False
        self.step_num = 0
        return self._return_observation()

    @property
    def workspace_data_path(self) -> str:
        return f"{self.workspace_path}/data/"

    def _return_observation(self):
        with open(self.workspace_path + "/code/code.py", "r") as f:
            code = f.read()
        model_code = code.split("@MODEL_START@")[1].split("@MODEL_END@")[0]
        return {MemKey.CODE: model_code}

    @staticmethod
    def answer_parser(raw_response):
        return raw_response

    def is_complete(self):
        return self.done

    def step(self, action: str) -> tuple[dict, float, bool]:
        """Perform an action and return the next observation, reward, and done."""
        print(action)

        self.step_num += 1
        if self.step_num == self.max_steps:
            self.done = True

        return {}, 0, self.done
