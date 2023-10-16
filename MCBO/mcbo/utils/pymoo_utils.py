from typing import Dict, Optional, List, Callable

import numpy as np
import pandas as pd
import torch
from pymoo.config import Config

from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.constraints_utils import input_eval_from_origx, sample_input_valid_points

Config.warnings['not_compiled'] = False

from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.variable import Real, Integer, Choice, Binary, Variable

from mcbo.search_space import SearchSpace
from mcbo.search_space.params.bool_param import BoolPara
from mcbo.search_space.params.integer_param import IntegerPara
from mcbo.search_space.params.nominal_param import NominalPara
from mcbo.search_space.params.numeric_param import NumericPara
from mcbo.search_space.params.pow_param import PowPara
from mcbo.trust_region.tr_manager_base import TrManagerBase
from mcbo.utils.discrete_vars_utils import get_discrete_choices
from mcbo.utils.distance_metrics import hamming_distance


class PymooProblem(Problem):
    def __init__(self, search_space: SearchSpace):

        self.search_space = search_space

        vars: Dict[str, Variable] = {}

        for i, name in enumerate(search_space.params):
            param = search_space.params[name]
            if isinstance(param, NumericPara):
                vars[name] = Real(bounds=(param.lb, param.ub))
            elif isinstance(param, PowPara):
                vars[name] = Real(bounds=(
                    param.transform(param.param_dict.get('lb')).item(),
                    param.transform(param.param_dict.get('ub')).item()))
            elif isinstance(param, IntegerPara):
                vars[name] = Integer(bounds=(param.lb, param.ub))
            elif isinstance(param, (NominalPara,)):
                vars[name] = Choice(options=np.arange(len(param.categories)))
            elif isinstance(param, BoolPara):
                vars[name] = Binary()  # TODO: debug this
            else:
                raise Exception(
                    f' The Genetic Algorithm optimizer can only work with numeric,'
                    f' integer, nominal and ordinal variables. Not with {type(param)}')

        super().__init__(vars=vars, n_obj=1, n_ieq_constr=0)

    def pymoo_to_mcbo(self, x) -> pd.DataFrame:
        # Convert X to a dictionary compatible with pandas
        x_pd_dict = {}
        for i, var_name in enumerate(self.search_space.param_names):
            param = self.search_space.params[var_name]
            x_pd_dict[self.search_space.param_names[i]] = []
            for j in range(len(x)):
                val = x[j][var_name]
                if isinstance(param, (NominalPara,)):
                    val = param.categories[val]
                if isinstance(param, PowPara):
                    val = param.inverse_transform(torch.tensor([val])).item()
                x_pd_dict[self.search_space.param_names[i]].append(val)

        return pd.DataFrame(x_pd_dict)

    def mcbo_to_pymoo(self, x):
        x_pymoo = []
        for i in range(len(x)):
            x_pymoo.append({})
            for j, param_name in enumerate(self.search_space.param_names):
                val = x.iloc[i][param_name]
                param = self.search_space.params[param_name]
                if isinstance(param, NominalPara):
                    val = param.categories.index(val)
                if isinstance(param, PowPara):
                    val = param.transform(val).item()
                x_pymoo[i][param_name] = val

        return np.array(x_pymoo)

    def _evaluate(self, x, out, *args, **kwargs):
        pass


class GenericRepair(Repair):

    def __init__(self, search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 tr_manager: TrManagerBase, pymoo_problem: PymooProblem):
        self.search_space = search_space
        self.tr_manager = tr_manager
        self.input_constraints = input_constraints
        self.pymoo_problem = pymoo_problem

        self.nominal_dims = self.search_space.nominal_dims
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims

        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)

        self.inverse_mapping = [(self.numeric_dims + self.search_space.nominal_dims).index(i) for i in
                                range(self.search_space.num_dims)]

        if self.tr_manager is None:
            self.tr_centre = None
            self.tr_centre_numeric = None
            self.tr_centre_nominal = None
        else:
            self.tr_centre = self.tr_manager.center.unsqueeze(0)
            self.tr_centre_numeric = self.tr_centre[:, self.numeric_dims]
            self.tr_centre_nominal = self.tr_centre[:, self.nominal_dims]

        super(GenericRepair, self).__init__()

    def _reconstruct_x(self, x_numeric: torch.FloatTensor, x_nominal: torch.FloatTensor) -> torch.FloatTensor:
        return torch.cat((x_numeric, x_nominal), dim=1)[:, self.inverse_mapping]

    def _do(self, problem, x: np.ndarray, **kwargs):
        x_mcbo = self.pymoo_problem.pymoo_to_mcbo(x)

        # project points back to TR
        if self.tr_manager:

            x_normalised = self.search_space.transform(x_mcbo)

            x_numeric = x_normalised[:, self.numeric_dims]
            x_nominal = x_normalised[:, self.nominal_dims]

            # Repair numeric variables
            if len(self.tr_centre_numeric[0]) > 0:
                # project back to x_centre vicinity
                delta_numeric = x_numeric - self.tr_centre_numeric[0]
                mask = delta_numeric > self.tr_manager.radii['numeric']
                x_numeric[mask] = \
                (self.tr_centre_numeric[0] + self.tr_manager.radii['numeric']).repeat(len(x_numeric), 1)[mask]

                delta_numeric = self.tr_centre_numeric[0] - x_numeric
                mask = delta_numeric > self.tr_manager.radii['numeric']
                x_numeric[mask] = \
                (self.tr_centre_numeric[0] - self.tr_manager.radii['numeric']).repeat(len(x_numeric), 1)[mask]

            # Repair nominal variables
            if len(self.tr_centre_nominal[0]) > 1:

                # Calculate the hamming distance of all nominal variables from the trust region centre
                d_hamming = hamming_distance(self.tr_centre_nominal, x_nominal, False)

                nominal_valid = d_hamming <= self.tr_manager.radii['nominal']

                # repair all invalid samples
                for sample_num in range(len(nominal_valid)):
                    if not nominal_valid[sample_num]:
                        mask = x_nominal[sample_num] != self.tr_centre_nominal[0]
                        indices = np.random.choice([idx for idx, x in enumerate(mask) if x],
                                                   size=d_hamming[sample_num].item() - self.tr_manager.radii['nominal'],
                                                   replace=False)
                        x_nominal[sample_num, indices] = self.tr_centre_nominal[0, indices]

            x_normalised = self._reconstruct_x(x_numeric, x_nominal)
            x_mcbo = self.search_space.inverse_transform(x_normalised)

        # check input constraints validity
        input_constr_valid_points = input_eval_from_origx(
            x=x_mcbo,
            input_constraints=self.input_constraints
        )
        input_constr_invalid_inds = np.arange(len(x))[np.logical_not(np.all(input_constr_valid_points, axis=1))]

        # sample valid candidates to replace invalid ones
        if self.tr_manager is not None:
            def point_sampler(n_points: int):
                transf_points = sample_numeric_and_nominal_within_tr(
                    x_centre=self.tr_centre,
                    search_space=self.search_space,
                    tr_manager=self.tr_manager,
                    n_points=n_points,
                    numeric_dims=self.numeric_dims,
                    discrete_choices=self.discrete_choices,
                )
                return self.search_space.inverse_transform(transf_points)
        else:
            point_sampler = self.search_space.sample

        if len(input_constr_invalid_inds) > 0:
            # sample valid points
            cands = sample_input_valid_points(
                n_points=len(input_constr_invalid_inds),
                point_sampler=point_sampler,
                input_constraints=self.input_constraints,
                allow_repeat=False,
            )
            for i in range(len(cands)):
                ind = input_constr_invalid_inds[i]
                x_mcbo.iloc[ind:ind + 1] = cands.iloc[i]

        x_pymoo = self.pymoo_problem.mcbo_to_pymoo(x_mcbo)

        return x_pymoo
