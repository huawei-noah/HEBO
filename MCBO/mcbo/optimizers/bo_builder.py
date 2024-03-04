import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union

import math
import numpy as np
import torch

from mcbo.acq_funcs import acq_factory
from mcbo.acq_optimizers import AcqOptimizerBase
from mcbo.acq_optimizers.genetic_algorithm_acq_optimizer import GeneticAlgoAcqOptimizer
from mcbo.acq_optimizers.interleaved_search_acq_optimizer import InterleavedSearchAcqOptimizer
from mcbo.acq_optimizers.local_search_acq_optimizer import LsAcqOptimizer
from mcbo.acq_optimizers.message_passing_optimizer import MessagePassingOptimizer
from mcbo.acq_optimizers.mixed_mab_acq_optimizer import MixedMabAcqOptimizer
from mcbo.acq_optimizers.random_search_acq_optimizer import RandomSearchAcqOptimizer
from mcbo.acq_optimizers.simulated_annealing_acq_optimizer import SimulatedAnnealingAcqOptimizer
from mcbo.models import ModelBase, ExactGPModel, ComboEnsembleGPModel, LinRegModel, RandDecompositionGP
from mcbo.models.gp.kernel_factory import mixture_kernel_factory, kernel_factory
from mcbo.optimizers import BoBase
from mcbo.search_space import SearchSpace
from mcbo.trust_region import TrManagerBase
from mcbo.trust_region.casmo_tr_manager import CasmopolitanTrManager
from mcbo.utils.graph_utils import laplacian_eigen_decomposition

# ------ MODEL KWs -------------------
DEFAULT_MODEL_EXACT_GP_KERNEL_KWARGS: Dict[str, Any] = dict(
    numeric_kernel_name='mat52',
    numeric_kernel_use_ard=True,
    numeric_lengthscale_constraint=None,
    nominal_lengthscale_constraint=None,
    nominal_kernel_kwargs=None,
    numeric_kernel_kwargs=None,
)

DEFAULT_MODEL_EXACT_GP_KWARGS: Dict[str, Any] = dict(
    noise_prior=None,
    noise_constr=None,
    noise_lb=1e-5,
    pred_likelihood=True,
    optimizer='adam',
    lr=3e-2,
    num_epochs=100,
    max_cholesky_size=2000,
    max_training_dataset_size=1000,
    max_batch_size=5000,
    verbose=False
)

DEFAULT_MODEL_DIFF_GP_KWARGS: Dict[str, Any] = dict(
    n_models=10,
    noise_lb=1e-5,
    n_burn=0,
    n_burn_init=100,
    max_training_dataset_size=1000,
    verbose=False
)

DEFAULT_MODEL_LIN_REG_KWARGS = dict(
    order=2,
    estimator='sparse_horseshoe',
    a_prior=2,
    b_prior=1,
    sparse_horseshoe_threshold=.1,
    n_gibbs=1e3,
)

# ------ ACQ OPTIM KWs -------------------

DEFAULT_ACQ_OPTIM_IS_KWARGS = dict(
    n_iter=100,
    n_restarts=3,
    max_n_perturb_num=20,
    num_optimizer='adam',
    num_lr=1e-3,
    nominal_tol=100,
)

DEFAULT_ACQ_OPTIM_LS_KWARGS = dict(
    n_random_vertices=20000,
    n_greedy_ascent_init=20,
    n_spray=10,
    max_n_ascent=float('inf'),
)

DEFAULT_ACQ_OPTIM_SA_KWARGS = dict(
    num_iter=100,
    n_restarts=3,
    init_temp=1,
    tolerance=100,
)

DEFAULT_ACQ_OPTIM_GA_KWARGS = dict(
    ga_num_iter=500,
    ga_pop_size=100,
    cat_ga_num_parents=20,
    cat_ga_num_elite=10,
    cat_ga_store_x=False,
    cat_ga_allow_repeating_x=True,
)

DEFAULT_ACQ_OPTIM_MAB_KWARGS = dict(
    batch_size=1,
    max_n_iter=200,
    mab_resample_tol=500,
    n_cand=5000,
    n_restarts=5,
    num_optimizer='sgd',
    cont_lr=3e-3,
    cont_n_iter=100,
)

DEFAULT_ACQ_OPTIM_MP_KWARGS = dict(
    acq_opt_restarts=1,
    max_eval=-4
)

DEFAULT_ACQ_OPTIM_RS_KWARGS = dict(
    num_samples=300,
)


# ------ BoBuilder -------------------

@dataclass
class BoBuilder:
    model_id: str = "gp_to"
    acq_opt_id: str = "ga"
    acq_func_id: str = "ei"
    tr_id: Optional[str] = "basic"
    init_sampling_strategy: str = "uniform"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    acq_opt_kwargs: Dict[str, Any] = field(default_factory=dict)
    tr_kwargs: Dict[str, Any] = field(default_factory=dict)
    acq_func_kwargs: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def get_model(search_space: SearchSpace, model_id: str, **model_kwargs) -> ModelBase:
        if model_id in ["gp_to", "gp_o", "gp_hed", "gp_ssk"]:
            gp_kwargs = DEFAULT_MODEL_EXACT_GP_KWARGS.copy()
            kernel_kwargs = DEFAULT_MODEL_EXACT_GP_KERNEL_KWARGS.copy()
            kernel_kwargs.update(model_kwargs.get("default_kernel_kwargs", {}))

            if model_id == "gp_to":
                kernel_kwargs["nominal_kernel_name"] = "transformed_overlap"
                kernel_kwargs["nominal_kernel_use_ard"] = model_kwargs.get("nominal_kernel_use_ard", True)
            elif model_id == "gp_o":
                kernel_kwargs["nominal_kernel_name"] = "overlap"
                kernel_kwargs["nominal_kernel_use_ard"] = model_kwargs.get("nominal_kernel_use_ard", True)
            elif model_id == "gp_hed":
                kernel_kwargs["nominal_kernel_name"] = "hed"
                kernel_kwargs["nominal_kernel_use_ard"] = False

                kernel_kwargs["nominal_kernel_hed_num_embedders"] = model_kwargs.get(
                    "nominal_kernel_hed_num_embedders", 128
                )

                nominal_hed_base_kernel = kernel_factory(
                    kernel_name=model_kwargs.get('model_cat_hed_base_kernel_name', "mat52"),
                    active_dims=np.arange(kernel_kwargs["nominal_kernel_hed_num_embedders"]),
                    use_ard=model_kwargs.get("model_cat_hed_base_kernel_ard", True),
                    lengthscale_constraint=None,
                    outputscale_constraint=None
                )

                kernel_kwargs["nominal_kernel_kwargs"] = {
                    "hed_base_kernel": nominal_hed_base_kernel,
                    "hed_num_embedders": kernel_kwargs["nominal_kernel_hed_num_embedders"]
                }
            elif model_id == "gp_ssk":
                kernel_kwargs["nominal_kernel_name"] = "ssk"
                kernel_kwargs["nominal_kernel_use_ard"] = model_kwargs.get("nominal_kernel_use_ard", True)
                assert len(search_space.nominal_dims) > 0, "No nominal dims"
                alphabet = search_space.params[search_space.param_names[search_space.nominal_dims[0]]].categories

                for dim in search_space.nominal_dims:
                    categories = search_space.params[
                        search_space.param_names[search_space.nominal_dims[dim]]].categories
                    assert categories == alphabet, f"Should have same categories {categories} - {alphabet}"

                kernel_kwargs["nominal_kernel_kwargs"] = dict(
                    seq_length=search_space.num_dims,
                    alphabet_size=len(alphabet),
                    gap_decay=model_kwargs.get("ssk_gap_decay", 0.5),
                    match_decay=model_kwargs.get("ssk_gap_decay", 0.8),
                    max_subsequence_length=model_kwargs.get("ssk_gap_decay", 3),
                    normalize=model_kwargs.get("ssk_normalize", True),
                )

                gp_kwargs["max_batch_size"] = 50
            else:
                raise ValueError(model_id)

            kernel = mixture_kernel_factory(
                search_space=search_space,
                numeric_kernel_name=kernel_kwargs["numeric_kernel_name"],
                numeric_kernel_use_ard=kernel_kwargs["numeric_kernel_use_ard"],
                numeric_lengthscale_constraint=kernel_kwargs["numeric_lengthscale_constraint"],
                nominal_kernel_name=kernel_kwargs["nominal_kernel_name"],
                nominal_kernel_use_ard=kernel_kwargs["nominal_kernel_use_ard"],
                nominal_lengthscale_constraint=kernel_kwargs["nominal_lengthscale_constraint"],
                nominal_kernel_kwargs=kernel_kwargs["nominal_kernel_kwargs"],
                numeric_kernel_kwargs=kernel_kwargs["numeric_kernel_kwargs"],
            )

            gp_kwargs.update(model_kwargs.get("gp_kwargs", {}))
            model = ExactGPModel(
                search_space=search_space,
                num_out=1,
                kernel=kernel,
                dtype=model_kwargs["dtype"],
                device=model_kwargs["device"],
                **gp_kwargs
            )
        elif model_id == "gp_diff":
            gp_kwargs = DEFAULT_MODEL_DIFF_GP_KWARGS.copy()
            gp_kwargs.update(model_kwargs.get("gp_kwargs", {}))
            n_vertices, adjacency_mat_list, fourier_freq_list, fourier_basis_list = laplacian_eigen_decomposition(
                search_space=search_space, device=model_kwargs["device"])
            model = ComboEnsembleGPModel(
                search_space=search_space,
                fourier_freq_list=fourier_freq_list,
                fourier_basis_list=fourier_basis_list,
                n_vertices=n_vertices,
                adjacency_mat_list=adjacency_mat_list,
                dtype=model_kwargs["dtype"],
                device=model_kwargs["device"],
                **gp_kwargs
            )
        elif model_id == "lr_sparse_hs":
            lin_reg_kwargs = DEFAULT_MODEL_LIN_REG_KWARGS.copy()
            lin_reg_kwargs.update(model_kwargs.get("lin_reg_kwargs", {}))
            assert lin_reg_kwargs["estimator"] == 'sparse_horseshoe'
            model = LinRegModel(
                search_space=search_space,
                dtype=model_kwargs["dtype"],
                device=model_kwargs["device"],
                **lin_reg_kwargs
            )
        elif model_id in ["gp_rd", "gp_rdto", "gp_rdhed"]:
            gp_kwargs = DEFAULT_MODEL_EXACT_GP_KWARGS.copy()
            gp_kwargs["max_batch_size"] = 200
            gp_kwargs.update(model_kwargs.get("gp_kwargs", {}))
            kernel_kwargs = DEFAULT_MODEL_EXACT_GP_KERNEL_KWARGS.copy()
            if model_id == "gp_rd":
                kernel_kwargs["nominal_kernel_name"] = model_kwargs.get("nominal_kernel_name", "overlap")
            elif model_id == "gp_rdto":
                kernel_kwargs["nominal_kernel_name"] = model_kwargs.get("nominal_kernel_name", "transformed_overlap")
            elif model_id == "gp_rdhed":
                gp_kwargs["hed"] = True
                kernel_kwargs["nominal_kernel_name"] = None

            if model_id != "gp_rdhed":
                assert not gp_kwargs.get("hed", False), gp_kwargs

            kernel_kwargs["nominal_kernel_use_ard"] = model_kwargs.get("nominal_kernel_use_ard", True)
            kernel_kwargs.update(model_kwargs.get("default_kernel_kwargs", {}))

            model = RandDecompositionGP(
                search_space=search_space,
                num_out=1,
                dtype=model_kwargs["dtype"],
                device=model_kwargs["device"],
                base_kernel_num=kernel_kwargs["numeric_kernel_name"],
                num_lengthscale_constraint=kernel_kwargs["numeric_lengthscale_constraint"],
                base_kernel_nom=kernel_kwargs["nominal_kernel_name"],
                nom_lengthscale_constraint=kernel_kwargs["nominal_lengthscale_constraint"],
                base_kernel_nom_kwargs=kernel_kwargs["nominal_kernel_kwargs"],
                base_kernel_num_kwargs=kernel_kwargs["numeric_kernel_kwargs"],
                **gp_kwargs
            )
        else:
            raise ValueError(model_id)
        return model

    @staticmethod
    def get_acq_optim(search_space: SearchSpace, acq_optim_name: str, device: torch.device,
                      input_constraints: Optional[List[Callable[[Dict], bool]]],
                      obj_dims: Union[List[int], np.ndarray, None],
                      out_constr_dims: Union[List[int], np.ndarray, None],
                      out_upper_constr_vals: Optional[torch.Tensor],
                      **acq_optim_kwargs) -> AcqOptimizerBase:
        if acq_optim_name == "is":
            kwargs = DEFAULT_ACQ_OPTIM_IS_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = InterleavedSearchAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        elif acq_optim_name == "ls":
            kwargs = DEFAULT_ACQ_OPTIM_LS_KWARGS
            kwargs.update(acq_optim_kwargs)

            n_vertices, adjacency_mat_list, _, _ = laplacian_eigen_decomposition(
                search_space, device=device)
            acq_optim = LsAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                adjacency_mat_list=adjacency_mat_list,
                n_vertices=n_vertices,
                **kwargs
            )
        elif acq_optim_name == "sa":
            kwargs = DEFAULT_ACQ_OPTIM_SA_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = SimulatedAnnealingAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        elif acq_optim_name == "ga":
            kwargs = DEFAULT_ACQ_OPTIM_GA_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = GeneticAlgoAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        elif acq_optim_name == "mab":
            kwargs = DEFAULT_ACQ_OPTIM_MAB_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = MixedMabAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        elif acq_optim_name == "mp":
            kwargs = DEFAULT_ACQ_OPTIM_MP_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = MessagePassingOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        elif acq_optim_name == "rs":
            kwargs = DEFAULT_ACQ_OPTIM_RS_KWARGS
            kwargs.update(acq_optim_kwargs)
            acq_optim = RandomSearchAcqOptimizer(
                search_space=search_space,
                input_constraints=input_constraints,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                **kwargs
            )
        else:
            raise ValueError(acq_optim_name)
        return acq_optim

    @staticmethod
    def get_tr_manager(
            tr_id: Optional[str],
            search_space: SearchSpace,
            model: ModelBase,
            constr_models: List[ModelBase],
            obj_dims: Union[List[int], np.ndarray],
            out_constr_dims: Union[List[int], np.ndarray],
            out_upper_constr_vals: Optional[torch.Tensor],
            n_init: int,
            **tr_kwargs
    ) -> Optional[TrManagerBase]:
        if tr_id is None:
            return
        if tr_id == "basic":
            tr_model = copy.deepcopy(model)

            tr_restart_acq_name = tr_kwargs.get("restart_acq_name", 'lcb')
            tr_restart_n_cand = tr_kwargs.get("restart_n_cand")
            tr_min_num_radius = tr_kwargs.get("min_num_radius")
            tr_max_num_radius = tr_kwargs.get("max_num_radius")
            tr_init_num_radius = tr_kwargs.get("init_num_radius")
            tr_min_nominal_radius = tr_kwargs.get("min_nominal_radius")
            tr_max_nominal_radius = tr_kwargs.get("max_nominal_radius")
            tr_init_nominal_radius = tr_kwargs.get("init_nominal_radius")
            tr_radius_multiplier = tr_kwargs.get("radius_multiplier")
            tr_succ_tol = tr_kwargs.get("succ_tol")
            tr_fail_tol = tr_kwargs.get("fail_tol")
            tr_verbose = tr_kwargs.get("verbose", False)

            if tr_restart_n_cand is None:
                tr_restart_n_cand = min(100 * search_space.num_dims, 5000)
            else:
                assert isinstance(tr_restart_n_cand, int)
                assert tr_restart_n_cand > 0

            # Trust region for numeric variables (only if needed)
            if search_space.num_numeric > 0:
                if tr_min_num_radius is None:
                    tr_min_num_radius = 2 ** -5
                else:
                    assert 0 < tr_min_num_radius <= 1, \
                        ('Numeric variables are normalised to the interval [0, 1].'
                         ' Please specify appropriate Trust Region Bounds')
                if tr_max_num_radius is None:
                    tr_max_num_radius = 1
                else:
                    assert 0 < tr_max_num_radius <= 1, \
                        ('Numeric variables are normalised to the interval [0, 1].'
                         ' Please specify appropriate Trust Region Bounds')
                if tr_init_num_radius is None:
                    tr_init_num_radius = 0.8 * tr_max_num_radius
                else:
                    assert tr_min_num_radius < tr_init_num_radius <= tr_max_num_radius
                assert tr_min_num_radius < tr_init_num_radius <= tr_max_num_radius
            else:
                tr_min_num_radius = tr_init_num_radius = tr_max_num_radius = None

            # Trust region for nominal variables (only if needed)
            if search_space.num_nominal > 1:
                if tr_min_nominal_radius is None:
                    tr_min_nominal_radius = 1
                else:
                    assert 1 <= tr_min_nominal_radius <= search_space.num_nominal

                if tr_max_nominal_radius is None:
                    tr_max_nominal_radius = search_space.num_nominal
                else:
                    assert 1 <= tr_max_nominal_radius <= search_space.num_nominal

                if tr_init_nominal_radius is None:
                    tr_init_nominal_radius = math.ceil(0.8 * tr_max_nominal_radius)
                else:
                    assert tr_min_nominal_radius <= tr_init_nominal_radius <= tr_max_nominal_radius

                assert tr_min_nominal_radius < tr_init_nominal_radius <= tr_max_nominal_radius, (
                    tr_min_nominal_radius, tr_init_nominal_radius, tr_max_nominal_radius)
            else:
                tr_min_nominal_radius = tr_init_nominal_radius = tr_max_nominal_radius = None

            if tr_radius_multiplier is None:
                tr_radius_multiplier = 1.5

            if tr_succ_tol is None:
                tr_succ_tol = 3

            if tr_fail_tol is None:
                tr_fail_tol = 40

            tr_acq_func = acq_factory(acq_func_id=tr_restart_acq_name, **tr_kwargs.get("tr_restart_acq_kwargs", {}))

            tr_manager = CasmopolitanTrManager(
                search_space=search_space,
                model=tr_model,
                constr_models=constr_models,
                obj_dims=obj_dims,
                out_constr_dims=out_constr_dims,
                out_upper_constr_vals=out_upper_constr_vals,
                acq_func=tr_acq_func,
                n_init=n_init,
                min_num_radius=tr_min_num_radius,
                max_num_radius=tr_max_num_radius,
                init_num_radius=tr_init_num_radius,
                min_nominal_radius=tr_min_nominal_radius,
                max_nominal_radius=tr_max_nominal_radius,
                init_nominal_radius=tr_init_nominal_radius,
                radius_multiplier=tr_radius_multiplier,
                succ_tol=tr_succ_tol,
                fail_tol=tr_fail_tol,
                restart_n_cand=tr_restart_n_cand,
                verbose=tr_verbose,
                dtype=model.dtype,
                device=model.device
            )
        else:
            raise ValueError(tr_id)

        return tr_manager

    def build_bo(self, search_space: SearchSpace, n_init: int,
                 input_constraints: Optional[List[Callable[[Dict], bool]]] = None,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device('cpu'),
                 obj_dims: Union[List[int], np.ndarray, None] = None,
                 out_constr_dims: Union[List[int], np.ndarray, None] = None,
                 out_upper_constr_vals: Optional[np.ndarray] = None,
                 ) -> BoBase:
        """

        Args:
            search_space: search space
            n_init: number of initial points before building the surrogate
            input_constraints: constraints on the values of input variables
            obj_dims: dimensions in ys corresponding to objective values to minimize
            out_constr_dims: dimensions in ys corresponding to inequality constraints
            out_upper_constr_vals: values of upper bounds for inequality constraints
            dtype: torch type
            device: torch device

        Returns:

        """
        self.model_kwargs["dtype"] = dtype
        self.model_kwargs["device"] = device
        self.acq_opt_kwargs["dtype"] = dtype

        model = self.get_model(search_space=search_space, model_id=self.model_id, **self.model_kwargs)
        if out_constr_dims is None:
            out_constr_dims = []
            out_upper_constr_vals = []
        if obj_dims is None:
            obj_dims = [0]
        constr_models = [  # TODO: allow to have different models for the constraints
            copy.deepcopy(model) for _ in range(len(out_constr_dims))
        ]

        acq_func = acq_factory(self.acq_func_id, **self.acq_func_kwargs)
        acq_optim = self.get_acq_optim(
            search_space=search_space,
            acq_optim_name=self.acq_opt_id,
            device=device,
            input_constraints=input_constraints,
            obj_dims=acq_func.obj_dims,
            out_constr_dims=acq_func.out_constr_dims,
            out_upper_constr_vals=acq_func.out_upper_constr_vals,
            **self.acq_opt_kwargs
        )
        tr_manager = self.get_tr_manager(
            tr_id=self.tr_id,
            search_space=search_space,
            model=model,
            constr_models=constr_models,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
            n_init=n_init,
            **self.tr_kwargs
        )

        return BoBase(
            search_space=search_space,
            n_init=n_init,
            model=model,
            acq_func=acq_func,
            acq_optim=acq_optim,
            input_constraints=input_constraints,
            tr_manager=tr_manager,
            init_sampling_strategy=self.init_sampling_strategy,
            dtype=dtype,
            device=device,
            obj_dims=obj_dims,
            out_constr_dims=out_constr_dims,
            out_upper_constr_vals=out_upper_constr_vals,
        )

    def get_short_opt_name(self) -> str:
        short_opt_name = f"{self.model_id}--{self.acq_opt_id}--{self.acq_func_id}"

        if self.tr_id == "basic":
            short_opt_name += f"--TR"
        elif self.tr_id is not None:
            raise ValueError(self.tr_id)
        return short_opt_name


BO_ALGOS: Dict[str, BoBuilder] = dict(
    Casmopolitan=BoBuilder(model_id="gp_to", acq_opt_id="is", acq_func_id="ei", tr_id="basic"),
    BOiLS=BoBuilder(model_id="gp_ssk", acq_opt_id="is", acq_func_id="ei", tr_id="basic"),
    COMBO=BoBuilder(model_id="gp_diff", acq_opt_id="ls", acq_func_id="ei", tr_id=None),
    BODi=BoBuilder(model_id="gp_hed", acq_opt_id="is", acq_func_id="ei", tr_id=None),
    BOCS=BoBuilder(model_id="lr_sparse_hs", acq_opt_id="sa", acq_func_id="ts", tr_id=None),
    BOSS=BoBuilder(model_id="gp_ssk", acq_opt_id="ga", acq_func_id="ei", tr_id=None),
    CoCaBO=BoBuilder(model_id="gp_o", acq_opt_id="mab", acq_func_id="ei", tr_id=None),
    RDUCB=BoBuilder(model_id="gp_rd", acq_opt_id="mp", acq_func_id="addlcb", tr_id=None)
)
