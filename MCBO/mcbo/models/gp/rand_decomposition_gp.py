import copy
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import math
import networkx as nx
import numpy as np
import torch
from disjoint_set import DisjointSet
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, AdditiveKernel
from gpytorch.priors import Prior
from gpytorch.utils.errors import NanError, NotPSDError

from mcbo.models.gp import ExactGPModel
from mcbo.models.gp.kernels import DecompositionKernel
from mcbo.models.model_base import EnsembleModelBase
from mcbo.search_space import SearchSpace


class RandDecompositionGP(ExactGPModel):
    supports_cuda = True
    support_grad = True
    support_multi_output = True

    @property
    def name(self) -> str:
        name = "GPRD"
        if self.hed:
            kernname = f"HED-{self.base_kernel_num}"
        else:
            kernname = f"{self.base_kernel_num}_{self.base_kernel_nom}"
        name += f" ({kernname})"
        return name

    def __init__(
            self,
            search_space: SearchSpace,
            num_out: int,
            base_kernel_num: str = 'rbf',
            base_kernel_nom: str = 'overlap',
            base_kernel_num_kwargs: Optional[Dict[str, Any]] = None,
            base_kernel_nom_kwargs: Optional[Dict[str, Any]] = None,
            num_lengthscale_constraint: Optional[torch.nn.Module] = None,
            nom_lengthscale_constraint: Optional[torch.nn.Module] = None,
            noise_prior: Optional[Prior] = None,
            noise_constr: Optional[Interval] = None,
            noise_lb: float = 1e-5,
            pred_likelihood: bool = True,
            lr: float = 1e-3,
            num_epochs: int = 100,
            optimizer: str = 'adam',
            max_cholesky_size: int = 2000,
            max_training_dataset_size: int = 1000,
            max_batch_size: int = 200,
            verbose: bool = False,
            print_every: int = 10,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device('cpu'),
            random_tree_size: float = 0.2,
            hed: bool = False,
            hed_kwargs: Optional[Dict[str, Any]] = None,
            batched_kernel: bool = True
    ):
        self.search_space = search_space

        self.base_kernel_num = base_kernel_num
        self.base_kernel_num_kwargs = base_kernel_num_kwargs
        self.num_lengthscale_constraint = num_lengthscale_constraint

        self.base_kernel_nom = base_kernel_nom
        self.base_kernel_nom_kwargs = base_kernel_nom_kwargs
        self.nom_lengthscale_constraint = nom_lengthscale_constraint

        self.random_tree_size = random_tree_size
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims
        self.batched_kernel = batched_kernel
        self.hed = hed
        self.hed_kwargs = {} if hed_kwargs is None else hed_kwargs

        self.graph = self.get_random_graph()
        kernel = self.build_kernels(self.graph, restart_lengthscales=True)
        self.neg_variance_cliques = set()

        super(RandDecompositionGP, self).__init__(
            search_space=search_space,
            num_out=num_out,
            kernel=kernel,
            noise_prior=noise_prior,
            noise_constr=noise_constr,
            noise_lb=noise_lb, pred_likelihood=pred_likelihood,
            lr=lr, num_epochs=num_epochs,
            optimizer=optimizer,
            max_cholesky_size=max_cholesky_size,
            max_training_dataset_size=max_training_dataset_size,
            max_batch_size=max_batch_size,
            verbose=verbose,
            print_every=print_every,
            dtype=dtype,
            device=device
        )

    def partial_predict(self, x: torch.Tensor, clique: Tuple[int]) -> (torch.Tensor, torch.Tensor):
        num_points = len(x)

        if num_points < self.max_batch_size:
            # Evaluate all points at once
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
                x = x.to(device=self.device, dtype=self.dtype)
                pred = self.psd_error_handling_gp_partial_forward(x, clique)
                if self.pred_likelihood:
                    pred = self.likelihood(pred)
                mu_ = pred.mean.reshape(-1, self.num_out)
                var_ = pred.variance.reshape(-1, self.num_out)
        else:
            # Evaluate all points in batches
            mu_ = torch.zeros((len(x), self.num_out), device=self.device, dtype=self.dtype)
            var_ = torch.zeros((len(x), self.num_out), device=self.device, dtype=self.dtype)
            for i in range(int(np.ceil(num_points / self.max_batch_size))):
                x_ = x[i * self.max_batch_size: (i + 1) * self.max_batch_size].to(self.device, self.dtype)
                pred = self.psd_error_handling_gp_partial_forward(x_, clique)
                if self.pred_likelihood:
                    pred = self.likelihood(pred)
                mu_temp = pred.mean.reshape(-1, self.num_out)
                var_temp = pred.variance.reshape(-1, self.num_out)

                mu_[i * self.max_batch_size: (i + 1) * self.max_batch_size] = mu_temp
                var_[i * self.max_batch_size: (i + 1) * self.max_batch_size] = var_temp

        mu = mu_ * self.y_std.to(mu_) + self.y_mean.to(mu_) / len(self.graph)
        var = (var_ * self.y_std.to(mu_) ** 2)
        return mu, var.clamp(min=torch.finfo(var.dtype).eps)

    def psd_error_handling_gp_partial_forward(self, x: torch.Tensor, clique: Tuple[int]) -> torch.tensor:
        try:
            pred = self.gp_partial_forward(x, clique, self.gp)
        except (NotPSDError, NanError) as error_gp:
            if isinstance(error_gp, NotPSDError):
                error_type = "notPSD-error"
            elif isinstance(error_gp, NanError):
                error_type = "nan-error"
            else:
                raise ValueError(type(error_gp))
            reduction_factor_0 = .8
            reduction_factor = .8
            valid = False
            n_training_points = len(self.gp.train_inputs[0])
            while not valid and n_training_points > 1:
                warnings.warn(
                    f"--- {error_type} -> remove {(1 - reduction_factor) * 100:.2f}% of training points randomly and retry to predict ---")
                try:
                    gp = copy.deepcopy(self.gp)
                    n_training_points = len(gp.train_inputs[0])
                    filtre = np.random.choice(np.arange(n_training_points), replace=False,
                                              size=math.ceil(n_training_points * reduction_factor))
                    gp.train_inputs = (gp.train_inputs[0][filtre],)
                    gp._train_targets = gp._train_targets[filtre]
                    pred = self.gp_partial_forward(x, clique, gp)

                    valid = True
                except (NotPSDError, NanError) as inside_error_gp:
                    if isinstance(inside_error_gp, NotPSDError):
                        inside_error_type = "notPSD-error"
                    elif isinstance(inside_error_gp, NanError):
                        inside_error_type = "nan-error"
                    else:
                        raise ValueError(type(inside_error_gp))
                    reduction_factor = reduction_factor * reduction_factor_0
                    pass

            if not valid:
                raise error_gp
        return pred

    def gp_partial_forward(self, x: torch.tensor, clique: tuple, gp: ExactGPModel,
                           between_point_covariance: bool = True) -> torch.tensor:

        train_inputs = list(gp.train_inputs) if gp.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in [x]]
        full_inputs = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs, inputs):
            # Make sure the batch shapes agree for training/test data
            if batch_shape != train_input.shape[:-2]:
                batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))

        mean = gp.mean(*full_inputs) / len(self.graph)
        if self.batched_kernel:
            cov = self.kernel(*full_inputs, clique=clique, verbose=True)
        else:
            cov = self.kernel.kernels[self.clique_to_ix[tuple(sorted(clique))]](*full_inputs)
        partial_output = MultivariateNormal(mean, cov)
        partial_mean, partial_covar = partial_output.loc, partial_output.lazy_covariance_matrix

        if gp.prediction_strategy is None:
            gp(x[0].unsqueeze(0))

        # Determine the shape of the joint distribution
        batch_shape = partial_output.batch_shape
        joint_shape = partial_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - gp.prediction_strategy.train_shape[0], *tasks_shape])

        # Make the prediction
        with gpytorch.settings._use_eval_tolerance():
            predictive_mean, predictive_covar = gp.prediction_strategy.exact_prediction(partial_mean, partial_covar)

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
        if not between_point_covariance:
            # sometimes we only need variance without between point covariance (e.g. UCB), more numerically stable
            predictive_covar *= torch.eye(predictive_covar.shape[0])
        return MultivariateNormal(predictive_mean, predictive_covar)

    def get_random_graph(self) -> List[List[int]]:

        if self.hed:
            hed_num_embedders = self.hed_kwargs.get('hed_num_embedders', 128)
            size = self.search_space.num_dims + hed_num_embedders
        else:
            size = self.search_space.num_dims
        graph = nx.empty_graph(size)
        disjoint_set = DisjointSet()
        connections_made = 0
        while connections_made < min(size - 1, max(int(self.random_tree_size * size), 1)):
            edge_in = random.randint(0, size - 1)
            edge_out = random.randint(0, size - 1)

            if edge_in == edge_out or disjoint_set.connected(edge_out, edge_in):
                continue
            else:
                connections_made += 1
                graph.add_edge(edge_in, edge_out)
                disjoint_set.union(edge_in, edge_out)

        if self.hed:
            for dim in self.search_space.nominal_dims:
                graph.remove_node(dim)

        return list(nx.find_cliques(graph))

    def build_kernels(self, decomposition: List[List[int]], restart_lengthscales=True) -> Kernel:
        if self.batched_kernel:
            new_kernel = DecompositionKernel(
                decomposition=decomposition,
                base_kernel_num=self.base_kernel_num,
                base_kernel_kwargs_num=self.base_kernel_num_kwargs,
                base_kernel_nom=self.base_kernel_nom,
                base_kernel_kwargs_nom=self.base_kernel_nom_kwargs,
                search_space=self.search_space,
                num_lengthscale_constraint=self.num_lengthscale_constraint,
                nom_lengthscale_constraint=self.nom_lengthscale_constraint,
                hed=self.hed,
                hed_kwargs=self.hed_kwargs
            )

            # emprically better performance if we restart the lengthscales
            if not restart_lengthscales:
                old_lengthscales = self.kernel.lengthscale
                new_kernel._set_lengthscale(old_lengthscales)
        else:
            new_kernels = []
            self.clique_to_ix = dict()
            for ix, clique in enumerate(decomposition):
                new_kernels.append(
                    ScaleKernel(
                        RBFKernel(active_dims=clique, ard_num_dims=len(clique))
                    )
                )
                self.clique_to_ix[tuple(sorted(clique))] = ix

            new_kernel = AdditiveKernel(*new_kernels)

        return new_kernel

    def fit(self, x: torch.Tensor, y: torch.Tensor, fixed_graph: Optional[List[List[int]]] = None) -> List[float]:
        if len(self.neg_variance_cliques):
            warnings.warn(
                f"{round(len(self.neg_variance_cliques) / len(self.graph), 2) * 100}% of cliques had negative variance on last fit. This is likely due to numerical instabilities."
            )
        if fixed_graph is None:
            graph = self.get_random_graph()
        else:
            graph = fixed_graph
        self.graph = graph
        self.kernel = self.build_kernels(graph)
        self.gp = None
        self.neg_variance_cliques = set()

        return super(RandDecompositionGP, self).fit(x, y)


class RandEnsembleGPModel(EnsembleModelBase):

    def __init__(
            self,
            search_space: SearchSpace,
            num_out: int,
            base_kernel_num: str = 'rbf',
            base_kernel_nom: str = 'overlap',
            base_kernel_num_kwargs: Optional[Dict[str, Any]] = None,
            base_kernel_nom_kwargs: Optional[Dict[str, Any]] = None,
            num_lengthscale_constraint: Optional[torch.nn.Module] = None,
            nom_lengthscale_constraint: Optional[torch.nn.Module] = None,
            noise_prior: Optional[Prior] = None,
            noise_constr: Optional[Interval] = None,
            noise_lb: float = 1e-5,
            pred_likelihood: bool = True,
            lr: float = 1e-3,
            num_epochs: int = 100,
            optimizer: str = 'adam',
            max_cholesky_size: int = 2000,
            max_training_dataset_size: int = 1000,
            max_batch_size: int = 1000,
            verbose: bool = False,
            print_every: int = 10,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device('cpu'),
            random_tree_size: float = 0.2,
            n_models=1
    ):

        self.rand_gp = RandDecompositionGP(
            search_space=search_space,
            num_out=num_out,
            base_kernel_num=base_kernel_num,
            base_kernel_nom=base_kernel_nom,
            base_kernel_num_kwargs=base_kernel_num_kwargs,
            base_kernel_nom_kwargs=base_kernel_nom_kwargs,
            num_lengthscale_constraint=num_lengthscale_constraint,
            nom_lengthscale_constraint=nom_lengthscale_constraint,
            noise_prior=noise_prior,
            noise_constr=noise_constr,
            noise_lb=noise_lb,
            pred_likelihood=pred_likelihood,
            lr=lr,
            num_epochs=num_epochs,
            optimizer=optimizer,
            max_cholesky_size=max_cholesky_size,
            max_training_dataset_size=max_training_dataset_size,
            max_batch_size=max_batch_size,
            verbose=verbose,
            print_every=print_every,
            dtype=dtype,
            device=device,
            random_tree_size=random_tree_size
        )

        super(RandEnsembleGPModel, self).__init__(
            search_space, num_out, n_models, dtype, device
        )

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Optional[List[float]]:
        self.models = []
        for _ in range(self.num_models):
            self.rand_gp.fit(x, y, **kwargs)
            self.models.append(copy.deepcopy(self.rand_gp))

        return

    def predict(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        mu = torch.zeros((len(x), self.num_out, self.num_models)).to(x)
        var = torch.zeros((len(x), self.num_out, self.num_models)).to(x)

        for i, model in enumerate(self.models):
            mu[..., i], var[..., i] = model.predict(x, **kwargs)

        return mu, var
