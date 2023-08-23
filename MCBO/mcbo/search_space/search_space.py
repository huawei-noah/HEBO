# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from mcbo.search_space.params.bool_param import BoolPara
from mcbo.search_space.params.int_exponent_param import IntExponentPara
from mcbo.search_space.params.integer_param import IntegerPara
from mcbo.search_space.params.nominal_param import NominalPara
from mcbo.search_space.params.numeric_param import NumericPara
from mcbo.search_space.params.param import Parameter
from mcbo.search_space.params.permutation_param import PermutationPara
from mcbo.search_space.params.pow_integer_param import PowIntegerPara
from mcbo.search_space.params.pow_param import PowPara
from mcbo.search_space.params.sigmoid_param import SigmoidPara


class SearchSpace:
    def __init__(self, params: List[dict], dtype: torch.dtype = torch.float64):
        super(SearchSpace, self).__init__()

        self.dtype = dtype

        # Register all parameter types
        self.param_types = {}
        self.register_param_type('pow', PowPara)
        self.register_param_type('sigmoid', SigmoidPara)
        self.register_param_type('bool', BoolPara)
        self.register_param_type('num', NumericPara)
        self.register_param_type('int', IntegerPara)
        self.register_param_type('nominal', NominalPara)
        self.register_param_type('pow_int', PowIntegerPara)
        self.register_param_type('permutation', PermutationPara)
        self.register_param_type('int_exponent', IntExponentPara)

        # Storage for all parameters and their names
        self.params: Dict[str, Parameter] = {}
        self.param_names = []

        # Storage for continuous parameters
        self.cont_names = []
        self.cont_dims = []

        # Storage for discrete parameters
        self.disc_names = []
        self.disc_dims = []

        # Storage for nominal parameters
        self.nominal_names = []
        self.nominal_dims = []

        # Storage for ordinal parameters
        self.ordinal_names = []
        self.ordinal_dims = []

        # Storage for permutation parameters
        self.perm_names = []
        self.perm_dims = []
        self.all_perm_dims = []
        self.perm_lengths = {}
        self.perm_col_names = {}

        # All dataframe column names
        self.df_col_names = []

        # Parse all parameters
        self.parse(params)

        self.opt_ub = np.array([self.params[p].opt_ub for p in self.param_names])
        self.opt_lb = np.array([self.params[p].opt_lb for p in self.param_names])

        self.transfo_ub = np.array([self.params[p].transfo_ub for p in self.param_names])
        self.transfo_lb = np.array([self.params[p].transfo_lb for p in self.param_names])

        self.cont_lb = [self.params[p].param_dict["lb"] for p in self.cont_names]
        self.cont_ub = [self.params[p].param_dict["ub"] for p in self.cont_names]

    def register_param_type(self, type_name, para_class):
        """
        User can define their specific parameter type and register the new type
        using this function
        """
        self.param_types[type_name] = para_class

    def parse(self, params: List[Dict]):

        for param_dict in params:
            assert (param_dict.get('type') in self.param_types), \
                f"Type {param_dict.get('type')} is not a valid parameter type. Please choose one" \
                f" of {[name for name in self.param_types]}"
            param = self.param_types[param_dict.get('type')](param_dict, self.dtype)
            assert np.sum(
                param.is_cont + param.is_disc + param.is_ordinal + param.is_nominal + param.is_permutation) == 1, \
                'parameter can have only a single type'

            self.param_names.append(param.name)
            self.params[param.name] = param

            if param.is_cont:
                self.cont_names.append(param.name)
                self.df_col_names = self.df_col_names + [param.name]
            elif param.is_disc:
                self.disc_names.append(param.name)
                self.df_col_names = self.df_col_names + [param.name]
            elif param.is_ordinal:
                self.ordinal_names.append(param.name)
                self.df_col_names = self.df_col_names + [param.name]
            elif param.is_nominal:
                self.nominal_names.append(param.name)
                self.df_col_names = self.df_col_names + [param.name]
            elif param.is_permutation:
                self.perm_names.append(param.name)
                self.perm_lengths[param.name] = param.length
                self.perm_col_names[param.name] = [param.name + f'_{i}' for i in range(param.length)]
                self.df_col_names = self.df_col_names + self.perm_col_names[param.name]

            else:
                raise Exception(f'Unknown parameter type for parameter {param.name}')

        idx = 0
        for param in self.param_names:
            if self.params[param].is_cont:
                self.cont_dims.append(idx)
                idx += 1
            elif self.params[param].is_disc:
                self.disc_dims.append(idx)
                idx += 1
            elif self.params[param].is_ordinal:
                self.ordinal_dims.append(idx)
                idx += 1
            elif self.params[param].is_nominal:
                self.nominal_dims.append(idx)
                idx += 1
            elif self.params[param].is_permutation:
                self.perm_dims.append([i for i in range(idx, idx + self.params[param].length)])

        self.all_perm_dims = []
        for i in self.perm_dims:
            self.all_perm_dims = self.all_perm_dims + i

    def sample(self, num_samples=1):
        """
        df_suggest: suggested initial points
        """
        df = pd.DataFrame(columns=self.df_col_names)
        if num_samples > 0:
            for name in self.param_names:
                if self.params[name].is_permutation:
                    df[self.perm_col_names[name]] = self.params[name].sample(num_samples)
                else:
                    df[name] = self.params[name].sample(num_samples)
        return df

    def transform(self, data: pd.DataFrame) -> torch.Tensor:
        """
        input: pandas dataframe
        output: xn and xe
        transform data to be within [opt_lb, opt_ub]
        """

        x = torch.full((len(data), self.num_dims), np.nan, dtype=self.dtype)

        idx = 0
        for param in self.param_names:
            if self.params[param].is_permutation:
                x[:, idx:idx + self.params[param].length] = self.params[param].transform(
                    data[self.perm_col_names[param]].values.copy())
                idx += self.params[param].length
            else:
                x[:, idx] = self.params[param].transform(data[param].values.copy())
                idx += 1

        return x

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        """
        input: x and xe
        output: pandas dataframe
        """

        inv_dict = {}

        idx = 0
        for param in self.param_names:
            if self.params[param].is_permutation:
                permutations = self.params[param].inverse_transform(x[:, idx: idx + self.params[param].length])
                for i, col in enumerate(self.perm_col_names[param]):
                    inv_dict[col] = permutations[:, i]

                idx += self.params[param].length
            else:
                inv_dict[param] = self.params[param].inverse_transform(x[:, idx])
                idx += 1

        inv_x = pd.DataFrame.from_dict(inv_dict)
        if self.num_cont > 0:  # deal with numerical inconsistencies
            inv_x[self.cont_names] = inv_x[self.cont_names].clip(self.cont_lb, self.cont_ub)
        return inv_x

    @property
    def num_params(self):
        return len(self.param_names)

    @property
    def num_dims(self):
        return self.num_cont + self.num_disc + self.num_ordinal + self.num_nominal + self.num_permutation_dims

    @property
    def num_cont(self):
        return len(self.cont_names)

    @property
    def num_disc(self):
        return len(self.disc_names)

    @property
    def num_numeric(self):
        return self.num_cont + self.num_disc

    @property
    def num_nominal(self):
        return len(self.nominal_names)

    @property
    def num_ordinal(self):
        return len(self.ordinal_names)

    @property
    def num_permutation(self):
        return len(self.perm_names)

    @property
    def num_permutation_dims(self):
        num_dims = 0
        for param in self.perm_names:
            num_dims += self.perm_lengths[param]
        return num_dims

    @property
    def disc_lb(self):
        dist_num_lb = [self.params[p].opt_lb for p in self.disc_names]
        return dist_num_lb

    @property
    def nominal_lb(self):
        nominal_lb = [self.params[p].opt_lb for p in self.nominal_names]
        return nominal_lb

    @property
    def disc_ub(self):
        dist_num_ub = [self.params[p].opt_ub for p in self.disc_names]
        return dist_num_ub

    @property
    def ordinal_ub(self):
        ordinal_ub = [self.params[p].opt_ub for p in self.ordinal_names]
        return ordinal_ub

    @property
    def nominal_ub(self):
        nominal_ub = [self.params[p].opt_ub for p in self.nominal_names]
        return nominal_ub

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SearchSpaceSubSet(SearchSpace):

    def __init__(self,
                 search_space: SearchSpace,
                 cont_dims: bool = False,
                 disc_dims: bool = False,
                 nominal_dims: bool = False,
                 ordinal_dims: bool = False,
                 permutation_dims: bool = False,
                 dtype: torch.dtype = torch.float64):

        params = []
        for param_name in search_space.param_names:

            append = False

            if cont_dims and search_space.params[param_name].is_cont:
                append = True
            if disc_dims and search_space.params[param_name].is_disc:
                append = True
            if nominal_dims and search_space.params[param_name].is_nominal:
                append = True
            if ordinal_dims and search_space.params[param_name].is_ordinal:
                append = True
            if permutation_dims and search_space.params[param_name].is_permutation:
                append = True

            if append:
                params.append(search_space.params[param_name].param_dict)

        super(SearchSpaceSubSet, self).__init__(params, dtype)
