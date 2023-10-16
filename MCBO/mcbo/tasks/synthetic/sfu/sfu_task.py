from typing import Union, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from mcbo.tasks import TaskBase
from mcbo.tasks.synthetic.sfu.utils_sfu import SFU_FUNCTIONS, _sfu_search_space_params_factory, \
    default_sfu_params_factory


class SfuTask(TaskBase):
    @property
    def name(self) -> str:
        return self.sfu_instance_task.name

    def __init__(self, task_name: str, variable_type: Union[str, List[str]], num_dims: Union[int, List[int]],
                 lb: Union[float, np.ndarray], ub: Union[float, np.ndarray],
                 num_categories: Optional[Union[int, List[int]]] = None, **task_instance_kwargs):
        super().__init__()

        if isinstance(num_dims, int):
            function_num_dims = num_dims
        elif isinstance(num_dims, list):
            function_num_dims = sum(num_dims)
        else:
            raise Exception('Expect num_dims to be either an integer or a list of integers')

        sfu_function_params = default_sfu_params_factory(
            task_name=task_name, num_dims=function_num_dims,
            task_name_suffix=task_instance_kwargs.get("task_name_suffix", None)
        )

        if lb is not None:
            sfu_function_params['lb'] = lb
        else:
            lb = sfu_function_params['lb']
        if ub is not None:
            sfu_function_params['ub'] = ub
        else:
            ub = sfu_function_params['ub']

        self.variable_type = variable_type
        self.num_dims = num_dims
        self.lb = lb
        self.ub = ub
        self.num_categories: Optional[Union[int, List[int]]] = num_categories

        self.sfu_instance_task = SFU_FUNCTIONS[task_name](
            **sfu_function_params
        )

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return _sfu_search_space_params_factory(
            variable_type=self.variable_type,
            num_dims=self.num_dims,
            lb=self.lb,
            ub=self.ub,
            num_categories=self.num_categories
        )

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        return self.sfu_instance_task.evaluate(x=x)
