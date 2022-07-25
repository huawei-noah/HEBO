# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import pandas as pd
import torch
from torch import Tensor

from .numeric_param     import NumericPara
from .integer_param     import IntegerPara
from .pow_param         import PowPara
from .categorical_param import CategoricalPara
from .bool_param        import BoolPara
from .pow_integer_param import PowIntegerPara
from .int_exponent_param import IntExponentPara
from .step_int import StepIntPara

class DesignSpace:
    def __init__(self):
        self.para_types = {}
        self.register_para_type('num', NumericPara)
        self.register_para_type('pow', PowPara)
        self.register_para_type('pow_int', PowIntegerPara)
        self.register_para_type('int_exponent', IntExponentPara)
        self.register_para_type('int', IntegerPara)
        self.register_para_type('step_int', StepIntPara)
        self.register_para_type('cat', CategoricalPara)
        self.register_para_type('bool', BoolPara)
        self.paras         = {}
        self.para_names    = []
        self.numeric_names = []
        self.enum_names    = []

    @property
    def num_paras(self):
        return len(self.para_names)

    @property
    def num_numeric(self):
        return len(self.numeric_names)

    @property
    def num_categorical(self):
        return len(self.enum_names)

    def parse(self, rec):
        self.para_config = rec
        self.paras       = {}
        self.para_names  = []
        for item in rec:
            assert(item['type'] in self.para_types)
            param = self.para_types[item['type']](item)
            self.paras[param.name] = param
            if param.is_categorical:
                self.enum_names.append(param.name)
            else:
                self.numeric_names.append(param.name)
        self.para_names = self.numeric_names + self.enum_names
        assert len(self.para_names) == len(set(self.para_names)), "There are duplicated parameter names"
        return self

    def register_para_type(self, type_name, para_class):
        """
        User can define their specific parameter type and register the new type
        using this function
        """
        self.para_types[type_name] = para_class

    def sample(self, num_samples = 1):
        """
        df_suggest: suggested initial points
        """
        df = pd.DataFrame(columns = self.para_names)
        for c in df.columns:
            df[c] = self.paras[c].sample(num_samples)
        return df

    def transform(self, data : pd.DataFrame) -> (Tensor, Tensor):
        """
        input: pandas dataframe
        output: xc and xe
        transform data to be within [opt_lb, opt_ub]
        """
        xc = data[self.numeric_names].values.astype(float).copy()
        xe = data[self.enum_names].values.copy()
        for i, name in enumerate(self.numeric_names):
            xc[:, i] = self.paras[name].transform(xc[:, i])
        for i, name in enumerate(self.enum_names):
            xe[:, i] = self.paras[name].transform(xe[:, i])
        return torch.FloatTensor(xc), torch.LongTensor(xe.astype(int))

    def inverse_transform(self, x : Tensor, xe : Tensor) -> pd.DataFrame:
        """
        input: x and xe
        output: pandas dataframe
        """
        with torch.no_grad():
            inv_dict = {}
            for i, name in enumerate(self.numeric_names):
                inv_dict[name] = self.paras[name].inverse_transform(x.detach().double().numpy()[:, i])
            for i, name in enumerate(self.enum_names):
                inv_dict[name] = self.paras[name].inverse_transform(xe.detach().numpy()[:, i])
            return pd.DataFrame(inv_dict)

    @property
    def opt_lb(self):
        lb_numeric = [self.paras[p].opt_lb for p in self.numeric_names]
        lb_enum    = [self.paras[p].opt_lb for p in self.enum_names]
        return torch.tensor(lb_numeric + lb_enum)

    @property
    def opt_ub(self):
        ub_numeric = [self.paras[p].opt_ub for p in self.numeric_names]
        ub_enum    = [self.paras[p].opt_ub for p in self.enum_names]
        return torch.tensor(ub_numeric + ub_enum)
