"""
Test the robust BO
"""
import numpy as np
import torch

RND_SEED = 42
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)

from model_utils.input_transform import additional_xc_samples, add_noise
from problems.problem_factory import get_test_problem, TestFunctions
from models.robust_gp import RobustGPModel, RobustGP
from models.uncertain_gp import UncertainGP, KN_SKL, KN_EXPECTED_RBF
from models.ugp import UGP
from models.mmd_gp import MMDGP
from utils.tb_logger import OptLogger
import utils.commons as cm
from utils import visulaization as vis
import model_utils.model_common_utils as mcu
from utils import input_uncertainty as iu

import matplotlib.pyplot as plt
import botorch as bot
from tqdm.auto import tqdm
from functools import partial
from scipy import stats
import pandas as pd
from typing import Callable, Dict
import os
from torch import multiprocessing
import traceback
import copy

mp = multiprocessing.get_context('spawn')

OPTIMUM_BOV = 'best_observed_value'
OPTIMUM_BE = 'best_expectation'


def run_BO(prob: TestFunctions,
           minimization: bool,
           num_expectation_eval,
           raw_input_mean: [float, np.array],
           raw_input_std: [float, np.array],
           xc_sample_size: int, n_var: int,
           input_sampling_func: Callable,
           input_type: str,
           init_xc_raw: np.array,
           init_y,
           init_expected_y: np.array,
           model_name,
           model_cls,
           model_config: Dict,
           fit_cfg: Dict,
           pred_cfgs: Dict,
           opt_cfg: Dict,
           x_bounds: np.array,
           n_iter: int,
           batch_size: int,
           n_restarts: int,
           raw_samples: int,
           converge_thr: float,
           oracle_optimum,
           oracle_opt_x,
           plot_freq: int,
           trial_name: str,
           save_dir: str,
           **kwargs):
    """
    Run a BayesOpt
    """
    print(f"[pid{os.getpid()}] {trial_name} starts...")
    device = model_config['device']
    cm.set_gmem_usage(device, reserved_gmem=6)
    dtype = model_config['dtype']
    max_acq_opt_retries = kwargs.get('max_acq_opt_retries', 3)
    opt_hist = []
    ret_que = kwargs.get("return_queue", None)
    # robust_end_pos = kwargs.get("robust_end_pos", np.array([3, 3]).reshape(1, 2))
    # env = prob.kwargs['env']
    try:
        # Initialize the opt logger
        opt_logger = OptLogger(['E[y]', ], [1.0, ], constr_names=None, tag=trial_name,
                               result_save_freq=1000, minimization=minimization)

        # BO loop
        tr_xc_raw = init_xc_raw.copy()
        tr_y = init_y.copy()
        optimum_method = opt_cfg.get('optimum_method', OPTIMUM_BOV)
        opt_x, opt_py, opt_expected_y = find_optimum(
            OPTIMUM_BOV, None, prob, n_var, minimization, tr_xc_raw, tr_y, None,
            input_sampling_func, num_expectation_eval
        )
        opt_logger.tb_logger.add_scalar("opt_expected_y", opt_expected_y, global_step=0)
        opt_hist.append((0, init_xc_raw, init_y, init_expected_y, opt_x, opt_py, opt_expected_y))

        obj2opt = bot.acquisition.objective.ScalarizedPosteriorTransform(
            weights=torch.tensor([-1.0 if minimization else 1.0], dtype=dtype, device=device)
        )
        te_xc_raw, te_y = prob.mesh_coords, prob.mesh_vals
        te_xc_raw_ts = torch.tensor(te_xc_raw, dtype=dtype, device=device) \
            if te_xc_raw is not None else None
        for iter_i in tqdm(range(1, n_iter + 1), desc=f'{trial_name}',
                           dynamic_ncols=True, leave=True):
            # prepare train data
            tr_x_ts, tr_y_ts = mcu.prepare_data(
                input_type, n_var, raw_input_mean, raw_input_std, xc_sample_size,
                input_sampling_func,
                tr_xc_raw, tr_y, dtype=dtype, device=device
            )

            # build model
            model = RobustGPModel(model_cls, **model_config)
            model.post_initialize(tr_x_ts, tr_y_ts)

            # fit
            fit_cfg['fit_name'] = f"{trial_name}/iter{iter_i}"
            model.fit(**fit_cfg)

            # find current optimum
            opt_x, opt_py, opt_expected_y = find_optimum(
                optimum_method, model, prob, n_var, minimization,
                tr_xc_raw, tr_y, tr_x_ts, input_sampling_func, num_expectation_eval
            )
            regret = oracle_optimum - opt_expected_y if oracle_optimum is not None \
                else opt_expected_y
            dist_2_opt = np.linalg.norm(oracle_opt_x - opt_x) if oracle_opt_x is not None else None
            opt_logger.tb_logger.add_scalar("opt_expected_y", opt_expected_y, global_step=iter_i)
            opt_logger.tb_logger.add_scalar("regret", regret, global_step=iter_i)
            if dist_2_opt is not None:
                opt_logger.tb_logger.add_scalar("distance_2_opt", dist_2_opt, global_step=iter_i)
            tqdm.write(f'[{trial_name}] current opt E[y]: {opt_expected_y:.3f}')

            # optimize
            acq_opt_success = False
            n_acq_opt_retry = 0
            x_candidates, candidate_acq_vals, acq = None, None, None
            while not acq_opt_success:
                try:
                    # acq = bot.acquisition.UpperConfidenceBound(
                    #     model=model.model, beta=4.0, posterior_transform=obj2opt
                    # )
                    acq = bot.acquisition.analytic.ExpectedImprovement(
                        model.model, best_f=min(tr_y) if minimization else max(tr_y),
                        posterior_transform=obj2opt
                    )
                    x_candidates, candidate_acq_vals = bot.optim.optimize_acqf(
                        acq_function=acq,
                        bounds=x_bounds,
                        q=batch_size,
                        num_restarts=n_restarts,
                        raw_samples=raw_samples,
                        options={"maxiter": 500},
                    )
                    acq_opt_success = True
                except Exception as e:
                    acq_opt_success = False
                    if n_acq_opt_retry >= max_acq_opt_retries:
                        raise e
                    else:
                        print('[Warn] Acq. optimization fails, try again:', e)
                        n_acq_opt_retry += 1

            # observe new values
            new_xc_raw = x_candidates.detach().cpu().numpy()
            new_xc_n = add_noise(new_xc_raw, sampling_func=input_sampling_func)
            new_y_n = prob.evaluate(new_xc_n)
            new_expected_y = prob.evaluate(
                additional_xc_samples(
                    new_xc_raw, num_expectation_eval, n_var, input_sampling_func
                ).reshape(-1, n_var),
            ).reshape(new_xc_raw.shape[0], -1).mean(axis=-1)

            # save to opt history
            opt_hist.append((iter_i, new_xc_raw, new_y_n, new_expected_y,
                             opt_x, opt_py, opt_expected_y))

            # visualize model prediction & acq.
            if n_var == 1 and (iter_i % plot_freq == 0 or iter_i == n_iter - 1):
                # eval
                with torch.no_grad():
                    te_pred = model.get_posterior(te_xc_raw_ts)
                    te_py = te_pred.mean.detach().cpu().numpy()
                    te_lcb, te_ucb = te_pred.mvn.confidence_region()
                    te_lcb, te_ucb = te_lcb.detach().cpu().numpy(), te_ucb.detach().cpu().numpy()
                    te_acq_vals = acq(te_xc_raw_ts.unsqueeze(-2)).detach().cpu().numpy()
                    xc_ls, xc_ls_str = mcu.get_kernel_lengthscale(model.model.covar_module)
                    xc_os, xc_os_str = mcu.get_kernel_output_scale(model.model.covar_module)
                    lkh_noise = model.likelihood.noise.item()

                # plot
                nrows, ncols, ax_scale = 2, 1, 1.5
                fig = plt.figure(figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
                axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=True, sharex=True)
                # plot model pred
                vis.plot_gp_predictions(
                    [(model_name, te_py.flatten(), te_ucb.flatten(), te_lcb.flatten(),
                      None, None), ],
                    te_xc_raw, prob.mesh_coords, prob.mesh_vals,
                    tr_xc_raw.flatten(), tr_y.flatten(),
                    fig=fig, ax=axes[0]
                )
                vis.scatter(
                    new_xc_raw.flatten(), new_y_n.flatten(), fig=fig, ax=axes[0],
                    label='new_p', marker='^', color='lime'
                )
                vis.scatter(
                    opt_x.flatten(), opt_py.flatten(), fig=fig, ax=axes[0],
                    label='optimum', marker='*', color='magenta'
                )
                axes[0].axvline(opt_x.flatten()[0], color='magenta', ls='--')
                axes[0].set_title(f"{model_name}\n"
                                  f"xc_ls={xc_ls_str}, xc_os={xc_os_str}, noise={lkh_noise:.3f}")
                axes[0].legend()

                # plot acq
                axes[1].plot(te_xc_raw.flatten(), te_acq_vals.flatten(),
                             label=f'{acq.__class__.__name__}')
                vis.scatter(
                    new_xc_raw.flatten(), candidate_acq_vals.detach().cpu().numpy().flatten(),
                    fig=fig, ax=axes[1],
                    label='new_p', marker='^', color='lime'
                )
                axes[1].legend()
                fig.tight_layout()
                opt_logger.tb_logger.add_figure('model_pred', fig, iter_i, close=True)

            # concat to the training data
            tr_xc_raw = np.concatenate([tr_xc_raw, new_xc_raw], axis=0)
            tr_y = np.concatenate([tr_y, new_y_n], axis=0)

            # # early stop
            # early_stop = (opt_expected_y <= converge_thr) if minimization \
            #     else (opt_expected_y >= converge_thr)
            # if early_stop:
            #     print(f'Current opt value {opt_expected_y:.3f} reaches '
            #           f'the convergence thr{converge_thr:.2f}, early quit!')
            #     break

    except Exception as e:
        # except ImportError as e:
        print(f"[Error] Trial: {trial_name} fails, its opt_hist might be incomplete.", e)
        print(traceback.format_exc())
    finally:
        cm.serialize_obj(opt_hist, f"{save_dir}{trial_name}.opt_hist")

    if ret_que is not None:
        ret_que.put((trial_name, opt_hist))
    else:
        return trial_name, opt_hist


def find_optimum(optimum_method, model, prob, n_var, minimization,
                 tr_xc_raw, tr_y, tr_x_ts, input_sampling_func, num_expectation_eval):
    opt_x, opt_py, opt_expected_y = None, None, None
    with torch.no_grad():
        if optimum_method == OPTIMUM_BOV:
            opt_ind = np.argmin(tr_y) if minimization else np.argmax(tr_y)
            opt_x = tr_xc_raw[opt_ind:opt_ind + 1]
            if model is not None and tr_x_ts is not None:
                opt_py = model.get_posterior(tr_x_ts[opt_ind:opt_ind + 1]) \
                    .mean.detach().cpu().numpy()
        elif optimum_method == OPTIMUM_BE:
            if model is None or tr_x_ts is None:
                raise ValueError("model and tr_x_ts should NOT be none.")
            observed_py = model.get_posterior(tr_x_ts).mean.detach().cpu().numpy()
            opt_ind = np.argmin(observed_py) if minimization else np.argmax(observed_py)
            opt_x = tr_xc_raw[opt_ind: opt_ind + 1]
            opt_py = observed_py[opt_ind: opt_ind + 1]
        else:
            raise ValueError('Unsupported optimum method:', optimum_method)

    opt_expected_y = prob.evaluate(
        additional_xc_samples(
            opt_x.reshape(-1, n_var), num_expectation_eval, n_var,
            input_sampling_func
        ).reshape(-1, n_var),
    ).flatten().mean()
    return opt_x, opt_py, opt_expected_y


# %%
if __name__ == "__main__":
    # %%
    # parameter
    import argparse

    parser = argparse.ArgumentParser(
        prog="compare_robust_opt", description='compare the optimization performance'
    )

    # optimization setup
    parser.add_argument('--init_samples', type=int, default=10, help='number of init samples')
    parser.add_argument('--n_iter', type=int, default=100, help='number of optimization iterations')
    parser.add_argument('--n_trial', type=int, default=10, help='number of trials for each optimization')

    # model setup
    parser.add_argument('--sampling_size', type=int, default=160, help='sampling size for MMD estimator')
    parser.add_argument('--sub_sampling_size', type=int, default=10, help='sub-sampling size for Nystrom')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--fit_with_scipy', type=bool, default=False, help='whether to use scipy fit')

    # problem setup
    parser.add_argument('--function_name', type=str, default='RKHS-S',
                        choices=['BumpedBowlHD', 'RKHS-S', 'CustomK'],
                        help='The name of the test function')
    parser.add_argument('--n_var', type=int, default=1, help='the input dimension of test function')
    parser.add_argument('--minimization', action="store_true",
                        help='the test function is a minimization problem')
    parser.add_argument('--n_evaluations', type=int, default=500,
                        help='the number of evaluations for true expectation')
    parser.add_argument('--input_distribution', type=str, default='step_chi2',
                        choices=['step_chi2', 'norm', 'gmm', 'beta', 'chi2', 'varying_beta', 'uniform', ],
                        required=True)
    parser.add_argument('--raw_input_std', type=float, default=0.01, help='the std of input uncertainty')

    # visualization setup
    parser.add_argument('--plot_acq', action="store_true",
                        help='plot the acquisition values')
    parser.add_argument('--separate_figures', action="store_true",
                        help='use a separate figure for each model')
    # run setup
    parser.add_argument('--debug_mode', action="store_true", help='run in debug mode')
    parser.add_argument('--use_gpu', action="store_true", help='use GPU')
    parser.add_argument('--use_double', action="store_true", help='use double precision')
    parser.add_argument('--use_multiprocess', action="store_true", help='run in parallel')

    args = parser.parse_args()

    # %%
    # general setup
    debug_mode = args.debug_mode
    use_gpu = args.use_gpu  # whether to use GPU by default
    use_multiprocess = args.use_multiprocess
    use_double_precision = args.use_double
    n_trial = 2 if debug_mode else args.n_trial
    n_iter = 5 if debug_mode else args.n_iter
    xc_sample_size = args.sampling_size  # sample size for each uncertain input sample
    sub_samp_size = args.sub_sampling_size
    naive_xc_sample_size = int((xc_sample_size * sub_samp_size) ** 0.5)
    init_samp_num = args.init_samples
    save_root_dir = "./results/"
    cm.ensure_dir_exist(save_root_dir)

    # %%
    # setup problem
    raw_fname = args.function_name
    unknown_optimum = False
    n_var = args.n_var  # variable dimension
    prob = get_test_problem(
        raw_fname, n_var=n_var, crpt_method="raw", mesh_sample_num=1000,
        input_range_offset=0, x_offset=0.0,
    )
    num_expectation_eval = args.n_evaluations

    # %%
    # prepare the initial data
    input_uncertainty_type = args.input_distribution
    raw_input_std = args.raw_input_std
    if input_uncertainty_type == "gmm":
        input_distrib = iu.GMMInputDistribution(
            n_components=2, mean=np.array([[-0.017], [0.03]]),
            covar=np.array([0.010 ** 2.0, 0.017 ** 2.0]),
            weight=np.array([0.5, 0.5])
        )
    elif input_uncertainty_type == 'beta':
        input_distrib = iu.ScipyInputDistribution(
            stats.beta(a=0.4, b=0.2, scale=raw_input_std), name='beta'
        )
    elif input_uncertainty_type == 'norm':
        if n_var > 1:
            input_distrib = iu.ScipyInputDistribution(
                stats.multivariate_normal(mean=[0., 0., 0.], cov=np.diag([1.5, 1.5, 2])),
                name="multivariate-normal"
            )
        else:
            input_distrib = iu.ScipyInputDistribution(
                stats.norm(loc=0, scale=raw_input_std), name="Gaussian", n_var=n_var
            )
    elif input_uncertainty_type == 'uniform':
        input_distrib = iu.ScipyInputDistribution(
            stats.uniform(loc=-0.15, scale=raw_input_std), name='uniform', n_var=1
        )
    elif input_uncertainty_type == 'concated_circular':
        input_distrib = iu.ConcatDistribution(
            distribs=[
                iu.Circular2Distribution(stats.uniform(), 'circular2d', radius=0.5),
                iu.ScipyInputDistribution(
                    stats.multivariate_normal(mean=[0, ] * (n_var - 2), cov=0.1 ** 2.0),
                    'mvn', n_var=n_var - 2
                )
            ],
            name="Circular2-Gaussian", n_var=n_var
        )
    else:
        raise ValueError('Unsupported input uncertainty type:', input_uncertainty_type)

    input_sampling_func = input_distrib.sample
    raw_input_std = input_distrib.estimated_var ** 0.5
    raw_input_mean = input_distrib.estimated_mean

    # find the exact and robust optimum
    if not unknown_optimum:
        all_expectations = prob.evaluate(
            additional_xc_samples(
                prob.mesh_coords, num_expectation_eval, n_var, input_sampling_func
            ).reshape(-1, n_var)
        ).reshape(prob.mesh_coords.shape[0], -1).mean(axis=-1)
        oracle_robust_opt_ind = np.argmin(all_expectations) if prob.minimization \
            else np.argmax(all_expectations)
        oracle_robust_optimum = all_expectations[oracle_robust_opt_ind]
        oracle_robust_opt_x = prob.mesh_coords[oracle_robust_opt_ind: oracle_robust_opt_ind + 1]
        oracle_exact_opt_ind = np.argmin(prob.mesh_vals) if prob.minimization \
            else np.argmax(prob.mesh_vals)
        oracle_exact_optimum = prob.mesh_vals.flatten()[oracle_exact_opt_ind]
        oracle_exact_opt_x = prob.mesh_coords[oracle_exact_opt_ind: oracle_exact_opt_ind + 1]
        converge_thr = np.quantile(all_expectations, 0.001) if prob.minimization \
            else np.quantile(all_expectations, 0.999)
    else:
        all_expectations = [None]
        oracle_exact_opt_ind = None
        oracle_exact_opt_x = None
        oracle_exact_optimum = None
        oracle_robust_opt_ind = None
        oracle_robust_opt_x = None
        oracle_robust_optimum = None
        converge_thr = oracle_exact_optimum
    print(f"{raw_fname}@{raw_input_mean}-{raw_input_std}: \n"
          f"[exact] opt_ind={oracle_exact_opt_ind}, opt_x={oracle_exact_opt_x}, "
          f"optimum={oracle_exact_optimum}, "
          f"E[y]={all_expectations[oracle_exact_opt_ind] if not unknown_optimum else None} \n"
          f"[robust] ind={oracle_robust_opt_ind}, opt_x: {oracle_robust_opt_x}, "
          f"optimum: {oracle_robust_optimum}")

    experiment_name = f'E-{raw_fname}-{input_distrib.name}-{cm.get_current_time_str("%m%d%H%M%S")}'
    save_dir = f"{save_root_dir}{experiment_name}/"
    cm.ensure_dir_exist(save_dir)

    print(f"==== {experiment_name}: init_samp_num={init_samp_num}, "
          f"input_mean={raw_input_mean}, input_std={raw_input_std}, "
          f"converge_thr={converge_thr}~{oracle_robust_optimum} ====")

    # %%
    # model setup
    default_dtype = torch.double if use_double_precision else torch.float32
    default_device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

    x_bounds = torch.tensor(prob.input_bounds).T
    default_model_cfg = {
        'noise_free': False, 'n_var': n_var, 'input_bounds': x_bounds, 'num_inputs': 1,
        'dtype': default_dtype, "device": default_device, 'input_type': mcu.INPUT_TYPE_NOISED,
        'raw_input_std': raw_input_std, "raw_input_mean": raw_input_mean,
    }
    default_fit_cfg = {'epoch_num': 1 if debug_mode else args.epoch, 'lr': args.lr, 'fit_with_scipy': args.fit_with_scipy,
                       'dtype': default_dtype, 'device': default_device, 'print_every': 50}
    default_pred_cfg = {'dtype': default_dtype, 'device': default_device}

    # model to run: model_name, model_cls, model_cfg, fit_cfg, pred_mode_cfgs, optimum_cfg
    model_candidates = [
        (
            f"MMDGP-nystrom", MMDGP,
            {
                **default_model_cfg,
                'latent_dim': 1,
                'hidden_dims': None,
                'weight_decay': 0.0,
                'input_type': mcu.INPUT_TYPE_SAMPLES,
                'estimator_name': 'nystrom',
                'num_inputs': 3,
                'input_sampling_func': input_sampling_func,
                'chunk_size': 512,
                'xc_sample_size': xc_sample_size,
                'sub_samp_size': sub_samp_size,
            },
            {
                **default_fit_cfg,
                'chunk_size': 512,
            },
            {
                'mode0': {**default_pred_cfg, },
            },
            {'optimum_method': OPTIMUM_BE}
        ),

        (
            "uGP", UGP,
            {
                **default_model_cfg,
                'latent_dim': 1,
                'hidden_dims': None,
                'weight_decay': 0.0,
                'input_type': mcu.INPUT_TYPE_SAMPLES,
                'estimator_name': 'integral',
                'num_inputs': 3,
                'input_sampling_func': input_sampling_func,
                'chunk_size': 512,
                'xc_sample_size': naive_xc_sample_size,
                'sub_samp_size': naive_xc_sample_size,
            },
            {
                **default_fit_cfg,
                'chunk_size': 512,
            },
            {
                'mode0': {**default_pred_cfg, },
            },
            {'optimum_method': OPTIMUM_BE}
        ),

        (
            "GP", RobustGP,
            {**default_model_cfg, },
            {
                **default_fit_cfg,
            },
            {'mode0': {**default_pred_cfg}},
            {'optimum_method': OPTIMUM_BOV}
        ),

        (
            "skl", UncertainGP,
            {
                **default_model_cfg,
                "kernel_name": KN_SKL,
                'input_type': mcu.INPUT_TYPE_DISTRIBUTION,
                'num_inputs': 3
            },
            {
                **default_fit_cfg,
            },
            {
                'mode0': default_pred_cfg,
            },
            {'optimum_method': OPTIMUM_BE}
        ),

        (
            "ERBF", UncertainGP,
            {
                **default_model_cfg,
                "kernel_name": KN_EXPECTED_RBF,
                'input_type': mcu.INPUT_TYPE_DISTRIBUTION,
                'num_inputs': 3,
            },
            {
                **default_fit_cfg,
            },
            {
                'mode0': default_pred_cfg
            },
            {'optimum_method': OPTIMUM_BE}
        ),

        # (
        #     f"MMDGP-raw-S", MMDGP,
        #     {
        #         **default_model_cfg,
        #         'latent_dim': 1,
        #         'hidden_dims': None,
        #         'weight_decay': 0.0,
        #         'input_type': mcu.INPUT_TYPE_SAMPLES,
        #         'estimator_name': 'empirical',
        #         'num_inputs': 3,
        #         'input_sampling_func': input_sampling_func,
        #         'chunk_size': 512,
        #         'xc_sample_size': naive_xc_sample_size,
        #         'sub_samp_size': naive_xc_sample_size,
        #     },
        #     {
        #         **default_fit_cfg,
        #         'chunk_size': 512,
        #     },
        #     {
        #         'mode0': {**default_pred_cfg, },
        #     },
        #     {'optimum_method': OPTIMUM_BE}
        # ),
        #
        # (
        #     f"MMDGP-raw-L", MMDGP,
        #     {
        #         **default_model_cfg,
        #         'latent_dim': 1,
        #         'hidden_dims': None,
        #         'weight_decay': 0.0,
        #         'input_type': mcu.INPUT_TYPE_SAMPLES,
        #         'estimator_name': 'empirical',
        #         'num_inputs': 3,
        #         'input_sampling_func': input_sampling_func,
        #         'chunk_size': 512,
        #         'xc_sample_size': xc_sample_size,
        #         'sub_samp_size': xc_sample_size,
        #     },
        #     {
        #         **default_fit_cfg,
        #         'chunk_size': 512,
        #     },
        #     {
        #         'mode0': {**default_pred_cfg, },
        #     },
        #     {'optimum_method': OPTIMUM_BE}
        # ),

    ]

    # %%
    # setup optimization
    batch_size = 1
    n_restarts = 10
    raw_samples = 512
    if debug_mode:
        pred_plot_freq = 1 if n_var == 1 else 9e99
    else:
        pred_plot_freq = 10 if n_var == 1 else 9e99

    # BO loop
    results = {}
    p_handles = []
    que = mp.Queue()
    n_proc = min(n_trial * len(model_candidates), torch.cuda.device_count() * 2)
    pool = mp.Pool(processes=n_proc)
    import platform

    if platform.system() == "Windows":
        mp.set_start_method('spawn')
    for m_name, m_cls, model_cfg, fit_cfg, pred_cfgs, opt_cfg in model_candidates:
        m_input_type = model_cfg['input_type']
        m_xc_sample_size = model_cfg.get('xc_sample_size', xc_sample_size)
        m_sub_samp_size = model_cfg.get('sub_samp_size', sub_samp_size)
        for trial_i in range(n_trial):
            # generate initial samples
            # init_sample_inds = np.random.choice(range(prob.mesh_coords.shape[0]), init_samp_num)
            # init_xc_raw = prob.mesh_coords[init_sample_inds]
            # init_xc_raw = np.random.uniform(0.8, 1.0, size=init_samp_num).reshape(-1, 1)
            init_xc_raw = np.concatenate(
                [
                    np.random.uniform(l, u, init_samp_num).reshape(init_samp_num, -1)
                    for l, u in prob.input_bounds
                ],
                axis=1
            )
            trial_name = f'{experiment_name}_M-{m_name}_T-{trial_i}'
            init_xc_n = add_noise(init_xc_raw, sampling_func=input_sampling_func)
            init_y_n = prob.evaluate(init_xc_n)
            init_expected_y = prob.evaluate(
                additional_xc_samples(
                    init_xc_raw, num_expectation_eval, n_var, input_sampling_func
                ).reshape(-1, n_var),
            ).reshape(init_xc_raw.shape[0], -1).mean(axis=-1)

            # update trial configs
            trial_model_cfg = copy.deepcopy(model_cfg)
            trial_fit_cfg = copy.deepcopy(fit_cfg)
            trial_pred_cfgs = copy.deepcopy(pred_cfgs)
            trial_opt_cfg = copy.deepcopy(opt_cfg)
            trial_dtype = trial_fit_cfg.get('dtype', default_dtype)
            trial_device = trial_fit_cfg.get('device', default_device)
            # update device id if multiple GPUs are available
            if 'cuda' in trial_device.type and torch.cuda.device_count() > 1:
                trial_device = torch.device(
                    f"cuda:{torch.cuda.device_count() - 1 - (trial_i % torch.cuda.device_count())}"
                )
                trial_model_cfg['device'] = trial_device
                trial_fit_cfg['device'] = trial_device
                for _, _pc in trial_pred_cfgs.items():
                    _pc['device'] = trial_device
            # update tensor type and device
            trial_x_bounds = x_bounds.to(trial_dtype).to(trial_device)
            trial_model_cfg['input_bounds'] = trial_x_bounds

            # run BO jobs
            if use_multiprocess:
                # run in parallel
                ph = pool.apply_async(
                    run_BO,
                    args=(copy.deepcopy(prob), prob.minimization,
                          num_expectation_eval, raw_input_mean, raw_input_std, m_xc_sample_size,
                          n_var, input_sampling_func, m_input_type,
                          init_xc_raw, init_y_n, init_expected_y, m_name, m_cls,
                          trial_model_cfg, trial_fit_cfg, trial_pred_cfgs, trial_opt_cfg,
                          trial_x_bounds, n_iter, batch_size, n_restarts, raw_samples,
                          converge_thr, oracle_robust_optimum,
                          # prob.mesh_coords[oracle_robust_opt_ind],
                          oracle_robust_opt_x,
                          pred_plot_freq, trial_name, save_dir)
                )
                p_handles.append(ph)
            else:
                # run sequentially
                t_name, t_opt_hist = run_BO(
                    prob, prob.minimization,
                    num_expectation_eval, raw_input_mean, raw_input_std, m_xc_sample_size, n_var,
                    input_sampling_func, m_input_type,
                    init_xc_raw, init_y_n, init_expected_y, m_name, m_cls,
                    trial_model_cfg, trial_fit_cfg, trial_pred_cfgs, trial_opt_cfg,
                    trial_x_bounds, n_iter, batch_size, n_restarts, raw_samples,
                    converge_thr, oracle_robust_optimum,
                    oracle_robust_opt_x,
                    pred_plot_freq, trial_name, save_dir
                )
                results[t_name] = t_opt_hist

    # wait results
    if use_multiprocess:
        pool.close()
        for h in p_handles:
            trial_name, trial_opt_hist = h.get()
            results[trial_name] = trial_opt_hist

    hist_fp = f"{save_dir}{experiment_name}.hist"
    cm.serialize_obj(results, hist_fp)
    print(f"Experiment result is saved at {hist_fp}")

    # %%
    # compute statistic results
    print("Computing the stats results")
    STATS_REGRET = "Regret"
    STATS_DIST_2_OPT = "Distance to optimum"
    selected_model_names = [
        'MMDGP-nystrom',
        'GP',
        'skl',
        'ERBF',
        # 'MMDGP-raw-S',
        # 'MMDGP-raw-L',
        'uGP',
    ]
    model_zorders = {
        'GP': 10,
        'skl': 20,
        'ERBF': 30,
        # 'MMDGP-raw-S': 40,
        # 'MMDGP-raw-L': 45,
        'uGP': 50,
        'MMDGP-nystrom': 60,
    }
    stats_metrics = [
        (STATS_REGRET, -1),
        # (STATS_DIST_2_OPT, -3),
    ]
    stats_results = {}
    for s_name, s_val_ind in stats_metrics:
        trial_stats = {}
        for m_name in selected_model_names:
            # m_name = m_candidate[0]

            m_trials = {k: r for k, r in results.items() if m_name in k}
            if len(m_trials) == 0:
                print(f"skip {m_name}")
                continue

            m_trial_stats = pd.concat(
                [
                    pd.DataFrame([(h[0], h[s_val_ind]) for h in t_history],
                                 columns=['step_i', s_name]).set_index('step_i')
                    for t_history in m_trials.values()
                ],
                axis=1
            )
            if s_name == STATS_REGRET:
                m_trial_stats.ffill(inplace=True)
                m_trial_stats = m_trial_stats.cummin(axis=0) \
                    if prob.minimization else m_trial_stats.cummax()
                if oracle_exact_optimum is not None:
                    m_trial_stats = (m_trial_stats - oracle_robust_optimum) if prob.minimization \
                        else (oracle_robust_optimum - m_trial_stats)
            elif s_name == STATS_DIST_2_OPT:
                m_trial_stats.ffill(inplace=True)
                m_trial_stats = m_trial_stats.applymap(
                    lambda x: np.linalg.norm(x.reshape(1, -1) - oracle_robust_opt_x.reshape(1, -1))
                )
                m_trial_stats = m_trial_stats.cummin(axis=0)
            else:
                raise ValueError("Unsupported stats name:", s_name)
            trial_stats[m_name] = m_trial_stats
        stats_results[s_name] = trial_stats

    # visualize the stats
    separate_mean_std = True
    nrows, ncols, ax_scale = len(stats_results), 2 if separate_mean_std else 1, 1.0
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale)
    )
    for s_i, (s_name, stat_result) in enumerate(stats_results.items()):
        for i, (m_name, m_stat) in enumerate(stat_result.items()):
            _x = m_stat.index.values
            _mean = m_stat.mean(axis=1)
            _std = m_stat.std(axis=1)
            _ucb = _mean + _std * 2.0
            _lcb = _mean - _std * 2.0
            _zo = model_zorders.get(m_name, i)
            axes[s_i, 0].plot(
                _x, _mean, label=f'{m_name}',
                **{**vis.LINE_STYLES[i + 1], 'alpha': 1.0, 'zorder': _zo}
            )
            if separate_mean_std:
                axes[s_i, 1].plot(
                    _x, _std, label=f'{m_name}',
                    **{**vis.LINE_STYLES[i + 1], 'alpha': 1.0, 'zorder': _zo}
                )
            else:
                axes[s_i, 0].fill_between(_x, _lcb, _ucb, **{**vis.AREA_STYLE[i + 1], 'alpha': 0.4,
                                                             'zorder': _zo})

        axes[s_i, 0].set_xlabel('step')
        axes[s_i, 0].set_xlim(0, n_iter)
        axes[s_i, 0].set_ylabel(f"{s_name}(mean)")
        axes[s_i, 0].set_yscale('log', base=10)
        # axes[s_i, 0].set_ylim(-0.5, 4)
        axes[s_i, 0].legend(ncol=2, loc='upper center', fontsize=9)
        if separate_mean_std:
            axes[s_i, 1].set_xlabel('step')
            axes[s_i, 1].set_xlim(0, n_iter)
            axes[s_i, 1].set_ylabel(f"{s_name}(std)")
            axes[s_i, 1].set_yscale('log', base=10)
            axes[s_i, 1].legend(ncol=2, loc='upper center', fontsize=9)
    fig.tight_layout()
    result_png_fp = f"{save_dir}{experiment_name}.png"
    fig.savefig(result_png_fp)
    print(f"Result is saved at: {result_png_fp}")
