# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from hebo.design_space.design_space import DesignSpace


class AbstractOptimizer(ABC):
    support_parallel_opt = False
    support_constraint = False
    support_multi_objective = False
    support_combinatorial = False
    support_contextual = False

    def __init__(self, space: DesignSpace, csv_save_path: Optional[str] = None):
        """
        AbstractOptimizer constructor
        -----------------------------

        space (DesignSpace): The design space defining the bounds of the optimization problem.
        csv_save_path (Optional[str]): Path to a CSV file where results are saved upon each call to observe. Defaults to None, in which case no results are saved.
        """
        self.space = space

        # Ensure csv_save_path ends with .csv
        if isinstance(csv_save_path, str):
            if not csv_save_path.endswith(".csv"):
                csv_save_path += ".csv"

        self.csv_save_path = csv_save_path

    @abstractmethod
    def suggest(self, n_suggestions=1, fix_input: dict = None):
        """
        Perform optimisation and give recommendation using data observed so far
        ---------------------
        n_suggestions:  number of recommendations in this iteration

        fix_input:      parameters NOT to be optimized, but rather fixed, this
                        can be used for contextual BO, where you can set the contex as a design
                        parameter and fix it during optimisation
        """
        pass

    def observe(self, x: pd.DataFrame, y: np.ndarray):
        """
        Observe new data
        """
        # Save results
        if isinstance(self.csv_save_path, str):
            results = x.copy()
            results["y"] = y.copy()
            results.to_csv(self.csv_save_path)

        # Observe new data
        self.observe_new_data(x, y)

    @abstractmethod
    def observe_new_data(self, x: pd.DataFrame, y: np.ndarray):
        pass

    @property
    @abstractmethod
    def best_x(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def best_y(self) -> float:
        pass
