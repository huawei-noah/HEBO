import inspect
import os
import shutil
from typing import Dict, List

from agent.utils.utils import get_agent_root_dir

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
        self.reflection_strategy = kwargs.get('reflection_strategy', None)
        self.seed = kwargs.get('seed', 0)
        self.results_path = self.get_results_path(
            workspace_path=self.workspace_path,
            reflection_strategy=self.reflection_strategy,
            seed=self.seed,
        )
        self.code_dir = f"{self.results_path}/code"

    def reset(self, next_subtask: str | None = None) -> Dict[str, str]:
        if next_subtask is not None:
            self.episode_counter = int(next_subtask)

        assert os.path.exists(self.workspace_path)
        assert os.path.exists(self.workspace_data_path)
        assert os.path.exists(f"{self.workspace_path}/code")
        os.makedirs(f"{self.workspace_path}/results", exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        # copy code into seed dir, so we can run multiple seeds in parallel
        os.makedirs(f'{self.results_path}/code', exist_ok=True)
        shutil.copy(f"{self.workspace_path}/code/code.py", f"{self.results_path}/code/code.py")

        # ----------------------------------------------------
        # TODO find a better solution but this will work for now...
        with open(f"{self.code_dir}/code.py", "r") as f:
            content = f.read()

        corrected_path = os.path.join(get_agent_root_dir(), self.workspace_data_path)
        pattern = 'FILE_PATH = "'
        start_index = content.find(pattern)
        if start_index != -1:
            end_index = content.find('"', start_index + len(pattern))
            if end_index != -1:
                old_path = content[start_index + len(pattern):end_index]
                new_content = content[:start_index + len(pattern)] + corrected_path + content[end_index:]
                with open(f"{self.code_dir}/code.py", "w") as f:
                    f.write(new_content)
                print(f"Replaced '{old_path}' with '{corrected_path}' in {self.code_dir}/code.py", flush=True)
        else:
            print(f"No FILE_PATH variable found in {self.code_dir}/code.py", flush=True)
        # ----------------------------------------------------

        # copy utils function from third_party/hyperopt to workspace/code
        k_folds_cv_str = inspect.getsource(k_folds_cv)
        imports_str = "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import StratifiedKFold\n\n"
        k_folds_cv_str = imports_str + k_folds_cv_str
        with open(f"{self.code_dir}/utils.py", "w") as f:
            f.write(k_folds_cv_str)

        self.done = False
        self.step_num = 0
        return self._return_observation()

    @property
    def workspace_data_path(self) -> str:
        return f"{self.workspace_path}/data/"

    @staticmethod
    def get_results_path(workspace_path: str, reflection_strategy: str | None = None, seed: int = None) -> str:
        if reflection_strategy is None:
            results_path = os.path.join(workspace_path, "results", "no_reflection")
        else:
            results_path = os.path.join(workspace_path, 'results', reflection_strategy)
        if seed is not None:
            results_path = os.path.join(results_path, f"seed_{seed}")
        return results_path

    def get_reflection_strategy_prompt_file(self) -> str | None:
        if self.reflection_strategy is None:
            return None
        reflection_strategy_file = f'reflection_strategy/{self.reflection_strategy}.jinja'
        if (get_agent_root_dir() / 'src/agent/prompts/templates/hyperopt' / reflection_strategy_file).exists():
            return 'reflection_strategy/naive.jinja'
        else:
            raise ValueError(f'{self.reflection_strategy} has no corresponding jinja template')

    def _return_observation(self):
        with open(self.workspace_path + "/code/code.py", "r") as f:
            code = f.read()
        return {
            MemKey.CODE: code,
            MemKey.REFLECTION_STRATEGY_PROMPT: self.get_reflection_strategy_prompt_file(),
            MemKey.CONTINUE_OR_TERMINATE_BO: "Continue",
            MemKey.RESULTS_DIR: self.results_path,
            MemKey.CODE_DIR: self.code_dir,
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
