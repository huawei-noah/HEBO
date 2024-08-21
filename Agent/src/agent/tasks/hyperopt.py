import inspect
import os
from typing import Dict, List

from agent.memory import MemKey
from agent.tasks.tasks import Task
from agent.utils.hyperopt_utils import k_folds_cv


class HyperOpt(Task):
    def __init__(self, workspace_path: str, task_id: str, **kwargs):
        super().__init__(**kwargs)
        self.episode_counter = 0
        self.test_scores: List[float] = []
        self.workspace_path = os.path.join(workspace_path, task_id)
        self.done = False
        self.step_num = 0
        self.max_steps = 1
        self.id = 'hyperopt'
        self.reflection_strategy = kwargs.get('reflection_strategy', 'naive')

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        assert os.path.exists(self.workspace_path)
        assert os.path.exists(self.workspace_data_path)
        assert os.path.exists(f"{self.workspace_path}/code")
        os.makedirs(f"{self.workspace_path}/results", exist_ok=True)

        # copy utils function from third_party/hyperopt to workspace/code
        k_folds_cv_str = inspect.getsource(k_folds_cv)
        imports_str = "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import StratifiedKFold\n\n"
        k_folds_cv_str = imports_str + k_folds_cv_str
        with open(f"{self.workspace_path}/code/utils.py", "w") as f:
            f.write(k_folds_cv_str)

        self.done = False
        self.step_num = 0
        return self._return_observation()

    @property
    def workspace_data_path(self) -> str:
        return f"{self.workspace_path}/data/"

    def get_reflection_strategy_prompt_file(self) -> str:
        if self.reflection_strategy == 'naive':
            return 'reflection_strategy/naive.jinja'
        else:
            raise ValueError(f'{self.reflection_strategy} is not supported')

    def _return_observation(self):
        with open(self.workspace_path + "/code/code.py", "r") as f:
            code = f.read()
        return {
            MemKey.CODE: code,
            MemKey.REFLECTION_STRATEGY_PROMPT: self.get_reflection_strategy_prompt_file(),
            MemKey.CONTINUE_OR_TERMINATE_BO: "Continue",
        }

    def answer_parser(self, raw_response: str) -> str:
        """Return a parsed response."""
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
