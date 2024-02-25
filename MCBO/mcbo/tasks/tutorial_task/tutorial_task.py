from typing import List, Dict, Any

import numpy as np
import pandas as pd

from mcbo.tasks.task_base import TaskBase


class TutorialTask(TaskBase):
    op_converter = {'cos': np.cos, 'sin': np.sin, 'exp': np.exp}

    @property
    def name(self) -> str:
        return 'Custom Task'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        y = np.zeros((len(x), 1))  # will be filled with evaluations
        for ind in range(len(x)):
            x_ind = x.iloc[ind].to_dict()  # convert to a dictionary
            ops = [self.op_converter[x_ind[f'op{j}']] for j in range(3)]
            y[ind] = ops[0](x_ind['x0']) / (1 + ops[1](x_ind['x1'])) + ops[2](x_ind['x2'])
        return y

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        params = [{'name': f'op{i}', 'type': 'nominal', 'categories': ['cos', 'sin', 'exp']} for i in range(3)]
        params.extend([{'name': f'x{i}', 'type': 'num', 'lb': -1, 'ub': 1} for i in range(3)])
        return params