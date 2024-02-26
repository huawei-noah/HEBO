# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from typing import Optional, List, Callable, Dict, Union

import numpy as np
import pandas as pd
import torch
from pymoo.config import Config

from mcbo.utils.plot_resource_utils import COLORS_SNS_10, get_color

Config.warnings['not_compiled'] = False

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.algorithms.soo.nonconvex.ga import comp_by_cv_and_fitness
from pymoo.core.evaluator import Evaluator
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.core.termination import NoTermination
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.problems.static import StaticProblem
from pymoo.util.display.single import SingleObjectiveOutput

from mcbo.optimizers.optimizer_base import OptimizerNotBO
from mcbo.search_space.search_space import SearchSpace
from mcbo.trust_region.tr_manager_base import TrManagerBase
from mcbo.trust_region.tr_utils import sample_numeric_and_nominal_within_tr
from mcbo.utils.distance_metrics import hamming_distance
from mcbo.utils.pymoo_utils import PymooProblem, GenericRepair


class PymooMixedVariableGaWithRepair(GeneticAlgorithm):

    def __init__(self,
                 pop_size=50,
                 n_offsprings=None,
                 repair=NoRepair(),
                 tournament_selection: bool = True,
                 **kwargs):
        output = SingleObjectiveOutput()
        sampling = MixedVariableSampling()

        if tournament_selection:
            selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
        else:
            selection = RandomSelection()

        mating = MixedVariableMating(selection=selection, eliminate_duplicates=MixedVariableDuplicateElimination(),
                                     repair=repair)

        eliminate_duplicates = MixedVariableDuplicateElimination()
        survival = FitnessSurvival()

        super().__init__(pop_size=pop_size, n_offsprings=n_offsprings, sampling=sampling, mating=mating,
                         eliminate_duplicates=eliminate_duplicates, output=output, survival=survival, repair=repair,
                         **kwargs)


class PymooGeneticAlgorithm(OptimizerNotBO):
    color_1: str = get_color(ind=5, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return PymooGeneticAlgorithm.color_1

    @staticmethod
    def get_color() -> str:
        return PymooGeneticAlgorithm.get_color_1()

    @property
    def name(self) -> str:
        if self.fixed_tr_manager is not None:
            name = 'Tr-based Genetic Algorithm'
        else:
            name = 'Genetic Algorithm'
        return name

    @property
    def tr_name(self) -> str:
        return "no-tr"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 pop_size=50,
                 n_offsprings=None,
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 store_observations: bool = False,
                 tournament_selection: bool = True,
                 dtype: torch.dtype = torch.float64,
                 ):

        super(PymooGeneticAlgorithm, self).__init__(
            search_space=search_space,
            input_constraints=input_constraints,
            dtype=dtype,
            obj_dims=obj_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            out_constr_dims=out_constr_dims
        )

        self.store_observations = store_observations
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings
        self.fixed_tr_manager = fixed_tr_manager

        self._pymoo_pop = None
        self._x_queue = pd.DataFrame(index=range(0), columns=self.search_space.df_col_names, dtype=float)
        self._pymoo_x_queue = []
        self._pymoo_y = np.array([])

        # Used for algorithm initialisation
        self._pymoo_proxy_problem = PymooProblem(search_space=search_space)
        self.repair = GenericRepair(
            search_space=search_space,
            input_constraints=input_constraints,
            tr_manager=fixed_tr_manager,
            pymoo_problem=self._pymoo_proxy_problem
        )

        self._pymoo_ga = PymooMixedVariableGaWithRepair(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            repair=self.repair,
            tournament_selection=tournament_selection
        )
        self._pymoo_ga.setup(
            problem=self._pymoo_proxy_problem,
            termination=NoTermination()
        )

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:

        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        if len(self._x_queue) == 0:
            # ask the algorithm for the next solution to be evaluated
            self._pymoo_pop = self._pymoo_ga.ask()
            self._pymoo_x_queue = self._pymoo_pop.get("X")
            self._x_queue = self._pymoo_proxy_problem.pymoo_to_mcbo(x=self._pymoo_x_queue)

        if n_suggestions > len(self._x_queue):
            raise Exception(
                'n_suggestions is larger then the number of remaining samples in the current population. '
                'To avoid this, ensure that pop_size is a multiple of n_suggestions.')

        x_next.iloc[: n_suggestions] = self._x_queue.iloc[: n_suggestions]
        self._x_queue = self._x_queue.drop([i for i in range(n_suggestions)]).reset_index(drop=True)

        return x_next

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:

        if isinstance(y, torch.Tensor):
            y_torch = y
            y = y.cpu().numpy()
        else:
            y_torch = torch.tensor(y, dtype=self.dtype)

        x_transf = self.search_space.transform(x)

        # Append the data to the internal data buffer
        if self.store_observations:
            self.data_buffer.append(x_transf, y_torch)

        # update best fx
        self.update_best(x_transf=x_transf, y=y_torch)

        self._pymoo_y = np.concatenate((self._pymoo_y, y.flatten()))

        if len(self._pymoo_y) == len(self._pymoo_x_queue):
            static = StaticProblem(self._pymoo_proxy_problem, F=self._pymoo_y)
            Evaluator().eval(problem=static, pop=self._pymoo_pop)

            # returned the evaluated individuals which have been evaluated
            self._pymoo_ga.tell(infills=self._pymoo_pop)
            self._pymoo_y = np.array([])

    def restart(self):
        """
        Function used to restart the internal state of the optimizer between different runs on the same task.
        :return:
        """
        self._restart()

        self._pymoo_pop = None
        self._x_queue = pd.DataFrame(index=range(0), columns=self.search_space.df_col_names, dtype=float)
        self._pymoo_x_queue = []
        self._pymoo_y = np.array([])

        # Used for algorithm initialisation
        self._pymoo_proxy_problem = PymooProblem(self.search_space)

        self._pymoo_ga = PymooMixedVariableGaWithRepair(
            pop_size=self.pop_size,
            n_offsprings=self.n_offsprings,
            repair=self.repair
        )
        self._pymoo_ga.setup(
            self._pymoo_proxy_problem,
            termination=NoTermination()
        )

    def set_x_init(self, x: pd.DataFrame):
        """
        Function to set query points that should be suggested during random exploration

        :param x:
        :return:
        """

        self._x_queue = x
        self._pymoo_x_queue = self._pymoo_proxy_problem.mcbo_to_pymoo(x)
        self._pymoo_pop = Population.new("X", self._pymoo_x_queue)

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        """
        Function used to initialise an optimizer with a dataset of observations
        :param x:
        :param y:
        :return:
        """
        # Initialise the pymoo algorithm
        pop = Population.new("X", self._pymoo_proxy_problem.mcbo_to_pymoo(x))
        static = StaticProblem(self._pymoo_proxy_problem, F=y.flatten())
        Evaluator().eval(static, pop)
        self._pymoo_ga.tell(infills=pop)

        # Set best x and y
        self.update_best(x_transf=self.search_space.transform(data=x), y=torch.tensor(y, dtype=self.dtype))


class CategoricalGeneticAlgorithm(OptimizerNotBO):
    color_1: str = get_color(ind=8, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return PymooGeneticAlgorithm.color_1

    @staticmethod
    def get_color() -> str:
        return PymooGeneticAlgorithm.get_color_1()

    @property
    def name(self) -> str:
        if self.tr_manager is not None:
            name = 'Tr-based Genetic Algorithm'
        else:
            name = 'Genetic Algorithm'
        return name

    @property
    def tr_name(self) -> str:
        return "no-tr"

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 pop_size: int = 40,
                 num_parents: int = 20,
                 num_elite: int = 10,
                 store_observations: bool = True,
                 allow_repeating_suggestions: bool = False,
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 dtype: torch.dtype = torch.float64,
                 ):

        assert search_space.num_nominal == search_space.num_dims, \
            'Genetic Algorithm currently supports only nominal variables'

        super(CategoricalGeneticAlgorithm, self).__init__(
            search_space=search_space,
            dtype=dtype,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals
        )

        self.pop_size = pop_size
        self.num_parents = num_parents
        self.num_elite = num_elite
        self.store_observations = store_observations
        self.allow_repeating_suggestions = allow_repeating_suggestions
        if fixed_tr_manager is not None:
            assert 'nominal' in fixed_tr_manager.radii, 'Trust Region manager must contain' \
                                                        ' a radius for nominal variables'
            assert fixed_tr_manager.center is not None, 'Trust Region does not have a centre.' \
                                                        ' Call tr_manager.set_center(center) to set one.'
        self.tr_manager = fixed_tr_manager
        self.tr_center = None if fixed_tr_manager is None else fixed_tr_manager.center

        # Ensure that the number of elite samples is even
        if self.num_elite % 2 != 0:
            self.num_elite += 1

        assert self.num_parents >= self.num_elite, \
            "\n The number of parents must be greater than the number of elite samples"

        # Storage for the population
        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        # Initialising variables that will store elite samples
        self.x_elite = None
        self.y_elite = None

        # If there is a trust region manager, sample the initial population within a trust region of the centre

        self.x_queue = self.sample_input_valid_points(n_points=self.pop_size,
                                                      point_sampler=self.get_tr_point_sampler())
        if self.tr_manager is not None:
            self.x_queue.iloc[0:1] = self.search_space.inverse_transform(self.tr_center.unsqueeze(0))

        self.map_to_canonical = self.search_space.nominal_dims
        self.map_to_original = [self.map_to_canonical.index(i) for i in range(len(self.map_to_canonical))]

        self.lb = self.search_space.nominal_lb
        self.ub = self.search_space.nominal_ub

    def get_tr_point_sampler(self) -> Callable[[int], pd.DataFrame]:
        """
        Returns a function taking a number n_points as input and that returns a dataframe containing n_points sampled
        in the search space (within the trust region if a trust region is associated to self)
        """
        if self.tr_manager is not None:
            def point_sampler(n_points: int):
                # Sample points in the trust region of the new centre
                return self.search_space.inverse_transform(
                    sample_numeric_and_nominal_within_tr(
                        x_centre=self.tr_manager.center,
                        search_space=self.search_space,
                        tr_manager=self.tr_manager,
                        n_points=n_points,
                        numeric_dims=[],
                        discrete_choices=[],
                        max_n_perturb_num=0,
                        model=None,
                        return_numeric_bounds=False
                    )
                )
        else:
            # sample a random population
            point_sampler = self.search_space.sample

        return point_sampler

    def initialize(self, x: pd.DataFrame, y: np.ndarray):
        assert len(x) < self.pop_size, 'Initialise currently does not support len(x) > population_size'
        assert y.ndim == 2
        assert y.shape[1] == 1
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == self.search_space.num_dims

        x = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x.clone(), y.clone())

        # Add data to current trust region data
        self.x_pop = torch.cat((self.x_pop, x.clone()), axis=0)
        self.y_pop = torch.cat((self.y_pop, y.clone()), axis=0)

        # update best fx
        self.update_best(x_transf=x, y=y)

    def set_x_init(self, x: pd.DataFrame):
        self.x_queue = x

    def restart(self):
        self._restart()

        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        self.x_queue = self.sample_input_valid_points(
            n_points=self.pop_size,
            point_sampler=self.search_space.sample
        )

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        assert n_suggestions <= self.pop_size

        idx = 0
        n_remaining = n_suggestions
        x_next = pd.DataFrame(index=range(n_suggestions), columns=self.search_space.df_col_names, dtype=float)

        # Get points from current population
        if n_remaining and len(self.x_queue):
            n = min(n_remaining, len(self.x_queue))
            x_next.iloc[idx: idx + n] = self.x_queue.iloc[idx: idx + n]
            self.x_queue = self.x_queue.drop(self.x_queue.index[[i for i in range(idx, idx + n)]]).reset_index(
                drop=True)

            idx += n
            n_remaining -= n

        while n_remaining:
            self._generate_new_population()

            n = min(n_remaining, len(self.x_queue))
            x_next.iloc[idx: idx + n] = self.x_queue.iloc[idx: idx + n]
            self.x_queue = self.x_queue.drop(self.x_queue.index[[i for i in range(idx, idx + n)]]).reset_index(
                drop=True)

            idx += n
            n_remaining -= n

        return x_next

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:

        x_transf = self.search_space.transform(x)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=self.dtype)

        assert len(x_transf) == len(y)

        # Add data to all previously observed data
        if self.store_observations or (not self.allow_repeating_suggestions):
            self.data_buffer.append(x_transf, y)

        # Add data to current population
        self.x_pop = torch.cat((self.x_pop, x_transf.clone()), axis=0)
        self.y_pop = torch.cat((self.y_pop, y.clone()), axis=0)

        # update best fx
        self.update_best(x_transf=x_transf, y=y)

    def _generate_new_population(self):

        # Sort the current population
        indices = self.y_pop.flatten().argsort()
        x_sorted = self.x_pop[indices]
        y_sorted = self.y_pop[indices].flatten()

        # Normalise the objective function
        min_y = y_sorted[0]
        if min_y < 0:
            norm_y = y_sorted + abs(min_y)

        else:
            norm_y = y_sorted.clone()

        max_y = norm_y.max()
        norm_y = max_y - norm_y + 1

        # Calculate probability
        sum_norm_y = norm_y.sum()
        prob = norm_y / sum_norm_y
        cum_prob = prob.cumsum(dim=0)

        if (self.x_elite is None) and (self.y_elite is None):
            self.x_elite = x_sorted[:self.num_elite].clone()
            self.y_elite = y_sorted[:self.num_elite].clone().view(-1, 1)

        else:
            x_elite = torch.cat((self.x_elite.clone(), x_sorted[:self.num_elite].clone()))
            y_elite = torch.cat((self.y_elite.clone(), y_sorted[:self.num_elite].clone().view(-1, 1)))
            indices = np.argsort(y_elite.flatten())
            self.x_elite = x_elite[indices[:self.num_elite]]
            self.y_elite = y_elite[indices[:self.num_elite]]

        # Select parents
        parents = torch.full((self.num_parents, self.search_space.num_dims), fill_value=torch.nan, dtype=self.dtype)

        # First, append the best performing samples to the list of parents
        parents[:self.num_elite] = self.x_elite

        # Then append random samples to the list of parents. The probability of a sample being picked is
        # proportional to the fitness of a sample
        for k in range(self.num_elite, self.num_parents):
            index = np.searchsorted(cum_prob, np.random.random())
            assert index < len(x_sorted), (index, cum_prob)
            parents[k] = x_sorted[index].clone()

        # New population
        pop = torch.full((self.pop_size, self.search_space.num_dims), fill_value=torch.nan, dtype=self.dtype)

        # Second, perform crossover with the previously determined subset of all the parents
        # for k in range(self.num_elite, self.population_size, 2):
        for k in range(0, self.pop_size, 2):
            def point_sampler(n_points: int) -> pd.DataFrame:
                assert n_points <= 2, n_points
                r1 = np.random.randint(0, self.num_parents)
                r2 = np.random.randint(0, self.num_parents)
                pvar1 = parents[r1].clone()
                pvar2 = parents[r2].clone()

                # Constraint satisfaction with rejection sampling
                # constraints_satisfied = False
                # while not constraints_satisfied:
                ch1, ch2 = self._crossover(pvar1, pvar2)
                ch1, ch2 = ch1.unsqueeze(0), ch2.unsqueeze(0)

                _ch1, _ch2 = None, None

                # Mutate child 1
                done = False
                counter = 0
                x_observed = None
                if not self.allow_repeating_suggestions:
                    x_observed = self.data_buffer.x
                while not done:
                    _ch1 = self._mutate(ch1)
                    # Check if sample is already present in pop
                    if torch.logical_not((_ch1 == pop).all(axis=1)).all():
                        # Check if the sample was observed before
                        if not self.allow_repeating_suggestions:
                            if torch.logical_not((_ch1 == x_observed).all(axis=1)).all():
                                done = True
                        else:
                            if torch.logical_not((_ch1 == self.x_elite).all(axis=1)).all():
                                done = True
                    counter += 1

                    # If not possible to generate a sample that has not been observed before, sample a random point
                    if not done and counter == 100:
                        _ch1 = self.search_space.transform(
                            self.sample_input_valid_points(n_points=1,
                                                           point_sampler=self.get_tr_point_sampler()))
                        if not self.allow_repeating_suggestions:
                            while torch.logical_not((_ch1 == x_observed).all(axis=1)).all():
                                _ch1 = self.search_space.transform(
                                    self.sample_input_valid_points(n_points=1,
                                                                   point_sampler=self.get_tr_point_sampler()))
                        done = True

                # Mutate child 2
                done = False
                counter = 0
                while not done:
                    _ch2 = self._mutate(ch2)
                    # Check if sample is already present in X_queue or in X_elites
                    # Check if the sample was observed before
                    if not self.allow_repeating_suggestions:
                        if torch.logical_not((_ch2 == x_observed).all(axis=1)).all():
                            done = True
                    else:
                        if torch.logical_not((_ch2 == self.x_elite).all(axis=1)).all():
                            done = True
                    counter += 1

                    # If not possible to generate a sample that has not been observed before, perform crossover again
                    if not done and counter == 100:
                        _ch2 = self.search_space.transform(
                            self.sample_input_valid_points(n_points=1,
                                                           point_sampler=self.get_tr_point_sampler()))
                        if not self.allow_repeating_suggestions:
                            while torch.logical_not((_ch2 == x_observed).all(axis=1)).all():
                                _ch2 = self.search_space.transform(
                                    self.sample_input_valid_points(n_points=1,
                                                                   point_sampler=self.get_tr_point_sampler()))
                        done = True
                points = self.search_space.sample(num_samples=n_points)
                cands = self.search_space.inverse_transform(torch.cat([_ch1, _ch2]))
                if n_points == 2:
                    points = cands
                elif n_points == 1:
                    points.iloc[0:1] = cands.iloc[np.random.randint(0, 2)]

                return points

            pop[k:k + 2] = self.search_space.transform(
                self.sample_input_valid_points(n_points=2, point_sampler=point_sampler)
            )
        self.x_queue = self.search_space.inverse_transform(pop)

        self.x_pop = torch.zeros((0, self.search_space.num_dims), dtype=self.dtype)
        self.y_pop = torch.zeros((0, 1), dtype=self.dtype)

        return

    def _crossover(self, x1: torch.Tensor, x2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert self.search_space.num_nominal == self.search_space.num_dims, \
            'Current crossover can\'t handle permutations'

        x1_ = x1.clone()
        x2_ = x2.clone()

        if self.tr_manager is not None:

            idx = np.random.randint(low=1, high=self.search_space.num_dims - 1)

            x1_[:idx] = x2[:idx]
            x2_[:idx] = x1[:idx]

            d_x1 = hamming_distance(self.tr_center.unsqueeze(0), x1_.unsqueeze(0), False)[0]
            d_x2 = hamming_distance(self.tr_center.unsqueeze(0), x2_.unsqueeze(0), False)[0]

            if d_x1 > self.tr_manager.get_nominal_radius():
                # Project x1_ back to the trust region
                mask = x1_ != self.tr_center
                indices = np.random.choice([i for i, x in enumerate(mask) if x],
                                           size=d_x1.item() - self.tr_manager.get_nominal_radius(), replace=False)
                x1_[indices] = self.tr_center[indices]

            if d_x2 > self.tr_manager.get_nominal_radius():
                # Project x2_ back to the trust region
                mask = x2_ != self.tr_center
                indices = np.random.choice([i for i, x in enumerate(mask) if x],
                                           size=d_x2.item() - self.tr_manager.get_nominal_radius(), replace=False)
                x2_[indices] = self.tr_center[indices]

        else:
            # starts from 1 and end at num_dims - 1 to always perform a crossover
            idx = np.random.randint(low=1, high=self.search_space.num_dims - 1)

            x1_[:idx] = x2[:idx]
            x2_[:idx] = x1[:idx]

        return x1_, x2_

    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        assert self.search_space.num_nominal == self.search_space.num_dims, \
            'Current mutate can\'t handle permutations'
        assert x.ndim == 2, (x.shape, self.map_to_canonical)
        x_ = x.clone()[:, self.map_to_canonical]

        if self.tr_manager is not None:

            for i in range(len(x)):
                done = False
                while not done:
                    cand = x_[i].clone()
                    idx = np.random.randint(low=0, high=self.search_space.num_dims)
                    categories = np.array(
                        [j for j in range(int(self.lb[idx]), int(self.ub[idx]) + 1) if j != x[i, idx]])
                    cand[idx] = np.random.choice(categories)
                    dist_to_center = hamming_distance(self.tr_center.unsqueeze(0), cand.unsqueeze(0), False)
                    if dist_to_center <= self.tr_manager.radii['nominal']:
                        done = True
                        x_[i] = cand

        else:

            for i in range(len(x)):
                idx = np.random.randint(low=0, high=self.search_space.num_dims)
                categories = np.array([j for j in range(int(self.lb[idx]), int(self.ub[idx]) + 1) if j != x[i, idx]])
                x_[i, idx] = np.random.choice(categories)

        x_ = x_[:, self.map_to_original]

        return x_


class GeneticAlgorithm(OptimizerNotBO):
    """
    A Genetic Algorithm (GA) optimizer that determines which exact GA algorithm to use based on the variable types in
    the search space. If the search space contains only nominal variables, an elitist GA algorithm will be used. If the
    search space contains any other variable type combinations, the Mixed Variable GA from pymoo will be used (see
    https://pymoo.org/customization/mixed.html). On purely combinatorial problems, the elitist GA algorithm can
    sometimes outperform the Mixed Variable GA from pymoo by an order of magnitude. However, at the same time it can be
    approximately 50% slower.
    """
    color_1: str = get_color(ind=8, color_palette=COLORS_SNS_10)

    @staticmethod
    def get_color_1() -> str:
        return PymooGeneticAlgorithm.color_1

    @staticmethod
    def get_color() -> str:
        return PymooGeneticAlgorithm.get_color_1()

    @property
    def name(self) -> str:
        return self.backend_ga.name

    @property
    def tr_name(self) -> str:
        return self.backend_ga.tr_name

    def __init__(self,
                 search_space: SearchSpace,
                 input_constraints: Optional[List[Callable[[Dict], bool]]],
                 obj_dims: Union[List[int], np.ndarray, None],
                 out_constr_dims: Union[List[int], np.ndarray, None],
                 out_upper_constr_vals: Optional[torch.Tensor],
                 pop_size: int = 40,
                 pymoo_ga_n_offsprings: Optional[int] = None,
                 fixed_tr_manager: Optional[TrManagerBase] = None,
                 cat_ga_num_parents: int = 20,
                 cat_ga_num_elite: int = 10,
                 store_observations: bool = True,
                 cat_ga_allow_repeating_suggestions: bool = False,
                 pymoo_ga_tournament_selection: bool = True,
                 dtype: torch.dtype = torch.float64
                 ):

        super(GeneticAlgorithm, self).__init__(
            search_space=search_space,
            input_constraints=input_constraints,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            dtype=dtype,
        )

        assert len(self.out_constr_dims) == 0, "Do not support multi-obj / constraints yet"
        assert len(self.obj_dims) == 1, "Do not support multi-obj / constraints yet"

        if search_space.num_nominal == search_space.num_dims:
            self.backend_ga = CategoricalGeneticAlgorithm(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                out_constr_dims=out_constr_dims,
                pop_size=pop_size,
                num_parents=cat_ga_num_parents,
                num_elite=cat_ga_num_elite,
                store_observations=store_observations,
                allow_repeating_suggestions=cat_ga_allow_repeating_suggestions,
                fixed_tr_manager=fixed_tr_manager,
                dtype=dtype
            )

        else:
            self.backend_ga = PymooGeneticAlgorithm(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                pop_size=pop_size,
                n_offsprings=pymoo_ga_n_offsprings,
                fixed_tr_manager=fixed_tr_manager,
                store_observations=store_observations,
                tournament_selection=pymoo_ga_tournament_selection,
                dtype=dtype
            )

        # Overwrite some attributes to have direct access to them
        self.data_buffer = self.backend_ga.data_buffer

    def method_suggest(self, n_suggestions: int = 1) -> pd.DataFrame:
        return self.backend_ga.method_suggest(n_suggestions)

    def method_observe(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self.backend_ga.observe(x=x, y=y)
        self._best_x = self.backend_ga._best_x
        self.best_y = self.backend_ga.best_y

    def restart(self) -> None:
        self.backend_ga.restart()

    def set_x_init(self, x: pd.DataFrame) -> None:
        self.backend_ga.set_x_init(x)

    def initialize(self, x: pd.DataFrame, y: np.ndarray) -> None:
        self.backend_ga.initialize(x, y)
