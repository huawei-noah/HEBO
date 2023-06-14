# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os.path
import warnings

import pandas as pd
from mcbo.utils.general_utils import save_w_pickle
from typing import Dict, Any

# TODO 'Result logger supports n_suggests==1 for now'
class ResultsLogger:

    def __init__(self):
        self.columns = ["Eval Num", 'f(x)', 'f(x*)', 'Elapsed Time']
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
            warnings.warn(f"X and Y are saved in different directories: {save_x_path}, {save_y_path}")

        pd.DataFrame(self.data, columns=self.columns).to_csv(save_y_path, index=False)
        save_w_pickle(self.x_data, save_x_path)

    def restart(self):
        self.data = []
        self.x_data = {}
