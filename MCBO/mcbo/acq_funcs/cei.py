# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
from typing import Union, List, Optional

import torch
from torch.distributions import Normal

from mcbo.acq_funcs import EI
from mcbo.acq_funcs.acq_base import ConstrAcqBase
from mcbo.models import ModelBase


class CEI(ConstrAcqBase):
    """
    Constrained Expected Improvement: \prod_i Pr(c_i(x) <= lambda_i) * EI(x)
    Formulation taken from `Bayesian Optimization with Inequality Constraints`, Gardner et al.
    All constraints are inequality constraints of type `c_i(x) <= lambda_i`
    The lambda_i should be given in the original output space (not warped, normalised,...)
    """

    def __init__(self, num_constr: int, augmented_ei: bool = False):
        super(CEI, self).__init__()
        self._num_constr = num_constr
        self.ei = EI(augmented_ei=augmented_ei)

    @property
    def name(self) -> str:
        if self.ei.augmented_ei:
            return "CAEI"
        return "CEI"

    @property
    def augmented_ei(self) -> bool:
        return self.ei.augmented_ei

    def evaluate(self,
                 x: torch.Tensor,
                 model: ModelBase,
                 constr_models: List[ModelBase],
                 out_upper_constr_vals: torch.Tensor,
                 best_y: Optional[Union[float, torch.Tensor]],
                 **kwargs
                 ) -> torch.Tensor:
        """
        Compute standard EI and multiply by probability of constraint satisfaction

        Args:
            out_upper_constr_vals: upper bound for constraints
            constr_models: model for each output associated to a constraint
            best_y: best observed objective value so far, if None then optimize feasibility probability
        """
        # Get `- EI(x)`
        if best_y is None:
            neg_ei = -1
        else:
            neg_ei = self.ei.evaluate(x=x, model=model, best_y=best_y, **kwargs)

        # Get Pr(c_i(x) <= lambda_i)
        if isinstance(constr_models, ModelBase):
            constr_models = [constr_models]

        assert len(constr_models) == len(
            out_upper_constr_vals), f"Nb. of constraint models: {len(constr_models)} | Nb. lambdas: {len(out_upper_constr_vals)}"

        feas_proba = torch.ones_like(neg_ei)

        for model_constr, lambda_constr in zip(constr_models, out_upper_constr_vals):
            mean, var = model_constr.predict(x)
            std = var.clamp_min(1e-9).sqrt().flatten()
            feas_proba *= Normal(loc=mean.flatten(), scale=std).cdf(lambda_constr.to(mean))

        return feas_proba * neg_ei

    @property
    def num_obj(self) -> int:
        return 1

    @property
    def num_constr(self) -> int:
        return self._num_constr
