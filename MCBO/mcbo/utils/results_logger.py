# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os.path
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Union, List, ClassVar

import numpy as np
import pandas as pd

from mcbo.utils.general_utils import save_w_pickle


# TODO 'Result logger supports n_suggests==1 for now'
class ResultsLogger:

    def __init__(self):
        self.columns = ['Eval Num', 'f(x)', 'f(x*)', 'Elapsed Time']
        self.data = []
        self.x_data = {}

    def append(self, eval_num: int, x: Dict[str, Any], y: float, y_star: float, elapsed_time: float):
        self.data.append([int(eval_num), y, y_star, elapsed_time])
        for k, v in x.items():
            if k not in self.x_data:
                self.x_data[k] = []
            self.x_data[k].append(v)

    def save(self, save_y_path: str, save_x_path: str):
        if save_x_path.split('.')[-1] != 'pkl':
            save_x_path = save_x_path + '.pkl'

        if save_y_path.split('.')[-1] != 'csv':
            save_y_path = save_y_path + '.csv'

        if os.path.dirname(save_y_path) != os.path.dirname(save_x_path):
            warnings.warn(f'X and Y are saved in different directories: {save_x_path}, {save_y_path}')

        os.makedirs(os.path.dirname(save_x_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_y_path), exist_ok=True)
        pd.DataFrame(self.data, columns=self.columns).to_csv(save_y_path, index=False)
        save_w_pickle(self.x_data, save_x_path)

    def restart(self):
        self.data = []
        self.x_data = {}


@dataclass
class ResultsLoggerV2:
    n_blackbox_outputs: int

    EVAL_NUM_KEY: ClassVar[str] = "Eval Num"
    BATCH_NUM_KEY: ClassVar[str] = "Batch Num"
    F_KEY: ClassVar[str] = "F"
    SUGGEST_TIME_KEY: ClassVar[str] = "Suggestion Time"
    OBS_TIME_KEY: ClassVar[str] = "Observation Time"
    EVAL_TIME_KEY: ClassVar[str] = "Evaluation Time"

    @staticmethod
    def get_f_i_key(i: int) -> str:
        return f"{ResultsLoggerV2.F_KEY}_{i}(x)"

    def __post_init__(self) -> None:
        self.columns = [self.EVAL_NUM_KEY, self.BATCH_NUM_KEY,
                        *[f'{self.get_f_i_key(i)}' for i in range(self.n_blackbox_outputs)],
                        self.SUGGEST_TIME_KEY, self.OBS_TIME_KEY, self.EVAL_TIME_KEY]
        self.data = pd.DataFrame(columns=self.columns)
        self.x_data = {}

    def append(self, eval_num: int, batch_num: int, x: Dict[str, Any], y: np.ndarray, suggest_time: float,
               observe_time: float, eval_time: float) -> None:

        @dataclass
        class Entry:
            eval_num: int
            batch_num: int
            fs: np.ndarray
            suggest_time: float
            observe_time: float
            eval_time: float

            def to_dict(self) -> Dict:
                return {
                    ResultsLoggerV2.EVAL_NUM_KEY: int(self.eval_num),
                    ResultsLoggerV2.BATCH_NUM_KEY: int(self.batch_num),
                    ResultsLoggerV2.SUGGEST_TIME_KEY: self.suggest_time,
                    ResultsLoggerV2.OBS_TIME_KEY: self.observe_time,
                    ResultsLoggerV2.EVAL_TIME_KEY: self.eval_time,
                    **{ResultsLoggerV2.get_f_i_key(i): self.fs[i] for i in range(len(self.fs))}
                }

        assert y.ndim == 1, y.shape
        new_entry = Entry(eval_num=eval_num, batch_num=batch_num, fs=y, suggest_time=suggest_time,
                          observe_time=observe_time, eval_time=eval_time)
        self.data = self.data.append(new_entry.to_dict(), ignore_index=True)
        for k, v in x.items():
            if k not in self.x_data:
                self.x_data[k] = []
            self.x_data[k].append(v)

    def extend(self, eval_nums: Union[np.ndarray, List[int]], batch_nums: Union[np.ndarray, List[int]],
               xs: List[Dict[str, Any]], ys: np.ndarray, suggest_times: Union[List[float], np.ndarray],
               observe_times: Union[List[float], np.ndarray], eval_times: Union[List[float], np.ndarray]) -> None:
        assert ys.ndim == 2 and ys.shape[1] == self.n_blackbox_outputs, ys.shape
        assert len(eval_nums) == len(batch_nums) == len(xs) == len(ys)
        assert len(eval_nums) == len(suggest_times) == len(observe_times) == len(eval_times)

        for i in range(len(eval_nums)):
            self.append(
                eval_num=eval_nums[i],
                batch_num=batch_nums[i],
                x=xs[i],
                y=ys[i],
                suggest_time=suggest_times[i],
                observe_time=observe_times[i],
                eval_time=eval_times[i]
            )

    def save(self, save_y_path: str, save_x_path: str):
        if save_x_path.split('.')[-1] != 'pkl':
            save_x_path = save_x_path + '.pkl'

        if save_y_path.split('.')[-1] != 'csv':
            save_y_path = save_y_path + '.csv'

        if os.path.dirname(save_y_path) != os.path.dirname(save_x_path):
            warnings.warn(f'X and Y are saved in different directories: {save_x_path}, {save_y_path}')

        os.makedirs(os.path.dirname(save_x_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_y_path), exist_ok=True)
        self.data.to_csv(save_y_path, index=False)
        save_w_pickle(self.x_data, save_x_path)

    def restart(self):
        self.data = pd.DataFrame(columns=self.columns)
        self.x_data = {}
