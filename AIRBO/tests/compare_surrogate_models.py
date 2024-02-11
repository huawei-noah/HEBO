# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

"""
This script compares the robust surrogate models under different input uncertainty
"""
import numpy as np
import torch
# RND_SEED = 42
# np.random.seed(RND_SEED)
# torch.manual_seed(RND_SEED)

from utils import commons as cm
from problems.problem_factory import get_test_problem
from problems.test_function import Problem
from models.robust_gp import RobustGPModel, RobustGP
from models.ugp import UGP
from models.uncertain_gp import UncertainGP, KN_SKL, KN_EXPECTED_RBF, KN_RBF
from models.mmd_gp import MMDGP
from model_utils import model_common_utils as mcu
from model_utils.input_transform import additional_xc_samples, add_noise, AdditionalFeatures
from utils import visulaization as vis
from utils import input_uncertainty as iu

import gpytorch as gpyt
import botorch as bot
from functools import partial
from scipy import stats
import matplotlib.pyplot as plt
import time
from torch import multiprocessing as mp
from typing import Callable, Dict

OPTIMUM_BOV = 'best_observed_value'
OPTIMUM_BE = 'best_expectation'


def evaluate_model(prob: Problem, n_var: int,
                   raw_input_mean: [float, np.array], raw_input_std: [float, np.array],
                   input_sampling_func: Callable, tr_xc_raw, tr_y,
                   input_bounds, m_name: str, m_cls: RobustGP, m_cfg: Dict, fit_cfg: Dict,
                   pred_cfgs: Dict, opt_cfg: Dict,
                   batch_size: int, n_restarts: int, raw_samples: int, max_acq_opt_retries: int,
                   num_expectation_eval: int):
    m_dtype = m_cfg['dtype']
    m_device = m_cfg['device']
    x_bounds = torch.tensor(input_bounds, dtype=m_dtype, device=m_device).T
    m_cfg['input_bounds'] = x_bounds
    # prepare data
    m_xc_sample_size = m_cfg.get('xc_sample_size', None)
    tr_x_ts, tr_y_ts = mcu.prepare_data(
        m_cfg['input_type'], n_var, raw_input_mean, raw_input_std,
        m_xc_sample_size, input_sampling_func,
        tr_xc_raw, tr_y, dtype=m_dtype, device=m_device
    )
    te_xc_raw, te_y = prob.mesh_coords, prob.mesh_vals
    te_xc_raw_ts = torch.tensor(te_xc_raw, dtype=m_dtype, device=m_device)

    # build model
    model = RobustGPModel(m_cls, **m_cfg)
    model.post_initialize(tr_x_ts, tr_y_ts)

    # fit
    fit_cfg['fit_name'] = f"{m_name}"
    fit_t0 = time.time()
    model.fit(**fit_cfg)
    fit_time = time.time() - fit_t0
    print(f"[{m_name}] model fitting costs {fit_time:.2f} seconds")

    # optimize
    acq_opt_success = False
    n_acq_opt_retry = 0
    x_candidates, candidate_acq_vals, acq = None, None, None
    obj2opt = bot.acquisition.objective.ScalarizedPosteriorTransform(
        weights=torch.tensor([-1.0 if prob.minimization else 1.0],
                             dtype=m_dtype, device=m_device)
    )
    opt_t0 = time.time()
    while not acq_opt_success:
        try:
            acq = bot.acquisition.analytic.UpperConfidenceBound(
                model.model, beta=2 ** 0.5,
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
            candidate_acq_vals = candidate_acq_vals.detach().cpu().numpy()
            acq_opt_success = True
        except Exception as e:
            acq_opt_success = False
            if n_acq_opt_retry >= max_acq_opt_retries:
                raise e
            else:
                print('[Warn] Acq. optimization fails, try again:', e)
                n_acq_opt_retry += 1
    opt_time = time.time() - opt_t0
    print(f"[{m_name}] Acq. opt costs {opt_time:.2f} seconds.")

    # observe new values
    new_xc_raw = x_candidates.detach().cpu().numpy()
    new_xc_n = add_noise(new_xc_raw, sampling_func=input_sampling_func,
                         sampling_cfg={'x': new_xc_raw[..., 0]})
    new_y_n = prob.evaluate(new_xc_n)
    new_expected_y = prob.evaluate(
        additional_xc_samples(
            new_xc_raw, num_expectation_eval, n_var, input_sampling_func
        ).reshape(-1, n_var),
    ).reshape(new_xc_raw.shape[0], -1).mean(axis=-1)

    # test
    with torch.no_grad():
        pred_t0 = time.time()
        te_pred = model.get_posterior(te_xc_raw_ts)
        te_py = te_pred.mean.detach().cpu().numpy()
        te_lcb, te_ucb = te_pred.mvn.confidence_region()
        te_lcb, te_ucb = te_lcb.detach().cpu().numpy(), te_ucb.detach().cpu().numpy()
        te_acq_vals = acq(te_xc_raw_ts.unsqueeze(-2)).detach().cpu().numpy()
        xc_ls, xc_ls_str = mcu.get_kernel_lengthscale(model.model.covar_module)
        xc_os, xc_os_str = mcu.get_kernel_output_scale(model.model.covar_module)
        lkh_noise = model.likelihood.noise.item()
        pred_time = time.time() - pred_t0
        print(f"[{m_name}] model pred costs {pred_time:.2f} seconds.")

    return m_name, (xc_ls_str, xc_os_str, lkh_noise,
                    fit_time, opt_time, pred_time,
                    tr_xc_raw, tr_y, te_xc_raw, te_py, te_ucb, te_lcb, te_acq_vals,
                    new_xc_raw, candidate_acq_vals)


# %%
if __name__ == "__main__":
    # %%
    # parameter
    import argparse

    parser = argparse.ArgumentParser(prog="compare_surrogate_models",
                                     description='compare the modeling performances of the robust models')

    # model setup
    parser.add_argument('--sampling_size', type=int, default=160, help='sampling size for MMD estimator')
    parser.add_argument('--sub_sampling_size', type=int, default=10, help='sub-sampling size for Nystrom')
    parser.add_argument('--epoch', type=int, default=200, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--fit_with_scipy', action="store_true", help='use scipy to fit models')
    parser.add_argument('--tr_samples', type=int, default=40, help='number of training samples')

    # problem setup
    parser.add_argument('--function_name', type=str, default='RKHS-S', help='The name of the test function')
    parser.add_argument('--n_var', type=int, default=1, help='the input dimension of test function')
    parser.add_argument('--minimization', action="store_true",
                        help='the test function is a minimization problem')
    parser.add_argument('--n_evaluations', type=int, default=500, help='the number of evaluations for true expectation')
    parser.add_argument('--raw_input_std', type=float, default=0.01, help='the std of input uncertainty')
    parser.add_argument('--input_distribution', type=str, default='step_chi2',
                        choices=['step_chi2', 'norm', 'gmm', 'beta', 'chi2', 'varying_beta', 'uniform', ],
                        required=True)

    # run setup
    parser.add_argument('--use_gpu', action="store_true", help='use GPU')
    parser.add_argument('--use_double', action="store_true", help='use double precision')

    # visualization setup
    parser.add_argument('--plot_acq', action="store_true", help='plot the acquisition value')
    parser.add_argument('--separate_figures', action="store_true",
                        help='use a separate figure for each model')

    args = parser.parse_args()

    # %%
    # general setup
    save_dir = "./results/"
    cm.ensure_dir_exist(save_dir)
    use_gpu = args.use_gpu  # whether to use GPU by default
    use_double_precision = args.use_double
    xc_sample_size = args.sampling_size  # sample size for each uncertain input sample
    sub_samp_size = args.sub_sampling_size
    KME_sample_size = int((xc_sample_size * sub_samp_size) ** 0.5)
    empirical_MMD_samp_size = int((xc_sample_size * sub_samp_size) ** 0.5)
    print(f"xc_samp_size={xc_sample_size}, sub_samp_size={sub_samp_size}, "
          f"empirical_MMD_samp_size={empirical_MMD_samp_size}, KME_samp_size={KME_sample_size}")
    tr_samp_num = args.tr_samples
    plot_acq = args.plot_acq
    separate_figures = args.separate_figures

    raw_fname = args.function_name  # the benchmark function name
    n_var = args.n_var  # variable dimension
    minimization = args.minimization
    num_expectation_eval = args.n_evaluations

    default_dtype = torch.double if use_double_precision else torch.float
    default_device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

    raw_input_std = args.raw_input_std
    input_uncertainty_type = args.input_distribution
    if input_uncertainty_type == 'norm':
        input_distrib = iu.ScipyInputDistribution(
            stats.norm(loc=0.0, scale=raw_input_std) if n_var == 1 \
                else stats.multivariate_normal(mean=[0.0, 0.0],
                                               cov=np.diag([raw_input_std ** 2, raw_input_std ** 2])),
            name='Gaussian', n_var=n_var)
    elif input_uncertainty_type == 'gmm':
        input_distrib = iu.GMMInputDistribution(
            n_components=2, mean=np.array([[-0.017], [0.03]]),
            covar=np.array([0.010 ** 2.0, 0.017 ** 2.0]),
            weight=np.array([0.5, 0.5])
        )
    elif input_uncertainty_type == 'beta':
        input_distrib = iu.ScipyInputDistribution(
            stats.beta(a=0.4, b=0.2, scale=raw_input_std), name='beta', n_var=1
        )
    elif input_uncertainty_type == 'chi2':
        input_distrib = iu.ScipyInputDistribution(
            stats.chi2(df=3, scale=raw_input_std), name='chi2', n_var=1
        )
    elif input_uncertainty_type == 'step_chi2':
        input_distrib = iu.StepInputDistribution(
            distribs=[stats.chi2(df=0.5, scale=raw_input_std),
                      stats.chi2(df=7.0, scale=raw_input_std), ],
            name='StepChi2', n_var=1
        )
    elif input_uncertainty_type == 'varying_beta':
        input_distrib = iu.VaryingInputDistribution(
            rvs=stats.beta.rvs,
            name='varyingBeta', n_var=1
        )
    elif input_uncertainty_type == 'uniform':
        input_distrib = iu.ScipyInputDistribution(
            stats.uniform(loc=raw_input_std * -0.5, scale=raw_input_std), name="Uniform", n_var=n_var
        )
    else:
        raise ValueError('Unknown input uncertainty type:', input_uncertainty_type)

    input_sampling_func = input_distrib.sample
    if input_distrib.estimated_var is not None:
        raw_input_std = input_distrib.estimated_var ** 0.5
    raw_input_mean = input_distrib.estimated_mean

    # %%
    # setup problem
    prob = get_test_problem(
        raw_fname, n_var=n_var, crpt_method="raw", mesh_sample_num=512,
        input_range_offset=0, x_offset=0.0,
    )

    # %%
    # model configs
    x_bounds = torch.tensor(prob.input_bounds, dtype=default_dtype, device=default_device).T
    default_model_cfg = {
        'noise_free': False, 'n_var': n_var, 'input_bounds': prob.input_bounds, 'num_inputs': 1,
        'dtype': default_dtype, "device": default_device, 'input_type': mcu.INPUT_TYPE_NOISED,
        'raw_input_std': raw_input_std,
    }
    default_fit_cfg = {'epoch_num': args.epoch, 'lr': args.lr, 'fit_with_scipy': args.fit_with_scipy,
                       'dtype': default_dtype, 'device': default_device, 'print_every': 10,
                       'noise_free': False}
    default_pred_cfg = {'dtype': default_dtype, 'device': default_device}

    # model to run: model_name, model_cls, model_cfg, fit_cfg, pred_mode_cfgs, optimum_cfg
    model_candidates = [
        (
            f"MMDGP-raw({empirical_MMD_samp_size})", MMDGP,
            {
                **default_model_cfg,
                'latent_dim': 1,
                'hidden_dims': None,
                'weight_decay': 0.0,
                'input_type': mcu.INPUT_TYPE_SAMPLES,
                'estimator_name': 'empirical',
                'num_inputs': 3,
                'input_sampling_func': input_sampling_func,
                'chunk_size': 512,
                'xc_sample_size': empirical_MMD_samp_size,
                'sub_samp_size': empirical_MMD_samp_size,
                'noise_constr': gpyt.constraints.Interval(1e-4, 2e-1)
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
            f"MMDGP-raw({xc_sample_size})", MMDGP,
            {
                **default_model_cfg,
                'latent_dim': 1,
                'hidden_dims': None,
                'weight_decay': 0.0,
                'input_type': mcu.INPUT_TYPE_SAMPLES,
                'estimator_name': 'empirical',
                'num_inputs': 3,
                'input_sampling_func': input_sampling_func,
                'chunk_size': 512,
                'xc_sample_size': xc_sample_size,
                'sub_samp_size': xc_sample_size,
                'noise_constr': gpyt.constraints.Interval(1e-4, 2e-1)
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
            f"MMDGP-nystrom({xc_sample_size}/{sub_samp_size})", MMDGP,
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
                'noise_constr': gpyt.constraints.Interval(1e-4, 1e-1)
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
            f"uGP({KME_sample_size})", UGP,
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
                'xc_sample_size': KME_sample_size,
                'sub_samp_size': sub_samp_size,
                'noise_constr': gpyt.constraints.Interval(1e-4, 1e-1)
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
            f"uGP({xc_sample_size})", UGP,
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
                'xc_sample_size': xc_sample_size,
                'sub_samp_size': xc_sample_size,
                'noise_constr': gpyt.constraints.Interval(1e-4, 1e-1)
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
            {**default_model_cfg, 'noise_free': False},
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
    ]
    # %%
    # prepare train data
    # tr_xc_raw = np.concatenate(
    #     [
    #         np.random.uniform(l, u, tr_samp_num).reshape(tr_samp_num, -1)
    #         for l, u in prob.input_bounds
    #     ],
    #     axis=1
    # )
    # tr_xc_raw = np.concatenate(
    #     [tr_xc_raw, np.random.uniform(0.2, 0.3, 10).reshape(-1, 1)],
    #     axis=0
    # )
    # tr_xc_raw = np.arange(0.1, 0.9, 0.1).reshape(-1, 1)
    tr_xc_raw = np.array(
        [0.1, 0.14, 0.2, 0.42, 0.5, 0.66, 0.72, 0.85, 0.92]
    ).reshape(-1, 1)

    tr_samp_num = len(tr_xc_raw)
    tr_xc_n = add_noise(tr_xc_raw, sampling_func=input_sampling_func,
                        sampling_cfg={'x': tr_xc_raw})
    tr_y = prob.evaluate(tr_xc_n)

    # %%
    # train & test & vis
    results = {}
    batch_size = 1
    n_restarts = 10
    raw_samples = 512
    max_acq_opt_retries = 3
    n_workers = min(len(model_candidates), 10)
    if n_workers > 1:
        handles = []
        import platform

        # if platform.system() == "Windows":
        mp.set_start_method('spawn')
        with mp.Pool(processes=n_workers) as p:
            for m_i, (m_name, m_cls, m_cfg, fit_cfg, pred_cfgs, opt_cfg) in enumerate(
                    model_candidates):
                h = p.apply_async(
                    evaluate_model,
                    args=(
                        prob, n_var, raw_input_mean, raw_input_std, input_sampling_func,
                        tr_xc_raw, tr_y, prob.input_bounds,
                        m_name, m_cls, m_cfg, fit_cfg, pred_cfgs, opt_cfg,
                        batch_size, n_restarts, raw_samples, max_acq_opt_retries,
                        num_expectation_eval
                    )
                )
                handles.append(h)

            for h in handles:
                m_name, m_ret = h.get()
                results[m_name] = m_ret
    else:
        for m_i, (m_name, m_cls, m_cfg, fit_cfg, pred_cfgs, opt_cfg) in enumerate(model_candidates):
            m_name, m_ret = evaluate_model(
                prob, n_var, raw_input_mean, raw_input_std, input_sampling_func,
                tr_xc_raw, tr_y, prob.input_bounds,
                m_name, m_cls, m_cfg, fit_cfg, pred_cfgs, opt_cfg,
                batch_size, n_restarts, raw_samples, max_acq_opt_retries, num_expectation_eval
            )
            results[m_name] = m_ret
    # %%
    # visualize model prediction & acq.
    sel_models = [
        f"MMDGP-nystrom({xc_sample_size}/{sub_samp_size})",
        f"MMDGP-raw({empirical_MMD_samp_size})",
        f"MMDGP-raw({xc_sample_size})",
        f'uGP({KME_sample_size})',
        f'uGP({xc_sample_size})',
        'GP',
        'skl',
        'ERBF',
    ]
    if prob.n_var == 1:
        nrows = 2 if plot_acq else 1
        ncols, ax_scale = len(results), 1.0
        if not separate_figures:
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, sharex='row', sharey='row',
                squeeze=False, figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale)
            )
        else:
            fig, axes = None, None

        # plot
        m_i = 0
        for m_name in sel_models:
            m_result = results.get(m_name, None)
            if m_result is None:
                print(f"No result for {m_name}")
                continue

            (xc_ls_str, xc_os_str, lkh_noise, fit_time, opt_time, pred_time,
             tr_xc_raw, tr_y, te_xc_raw, te_py, te_ucb, te_lcb, te_acq_vals,
             new_xc_raw, candidate_acq_vals) = m_result
            # plot model pred
            if separate_figures:
                fig = plt.figure(figsize=(4 * 1, 3 * nrows))
                ax_pred = plt.subplot(111)
            else:
                ax_pred = axes[0, m_i]
            vis.plot_gp_predictions(
                [(m_name, te_py.flatten(), te_ucb.flatten(), te_lcb.flatten(),
                  None, None), ],
                te_xc_raw, prob.mesh_coords, prob.mesh_vals,
                tr_xc_raw.flatten(), tr_y.flatten(),
                fig=fig, ax=ax_pred
            )
            ax_pred.set_title(
                f"{m_name}\n"
                f"xc_ls={xc_ls_str}, xc_os={xc_os_str}, noise={lkh_noise:.3f}"
                # f"fit_t={fit_time:.3f}, opt_t={opt_time:.3f}, pred_t={pred_time:.3f}"
            )
            ax_pred.legend(ncol=2, loc='upper right')
            ax_pred.set_ylabel("Model prediction")
            fig.tight_layout()

            # plot acq
            if plot_acq:
                ax_acq = axes[1, m_i]
                ax_acq.plot(te_xc_raw.flatten(), te_acq_vals.flatten(),
                            label='acquisition')
                vis.scatter(
                    new_xc_raw.flatten(),
                    candidate_acq_vals.flatten(),
                    fig=fig, ax=ax_acq,
                    label='new_p', marker='^', s=50, color='g', zorder=100, alpha=0.6
                )
                ax_acq.axvline(new_xc_raw.flatten()[0], ls='--', color='g', alpha=0.5)
                ax_acq.set_ylabel("Acquisition function")
                ax_acq.legend(ncol=2)
            m_i += 1
        result_png_fp = f"{save_dir}model_comparison_{raw_fname}_{input_uncertainty_type}.png"
        fig.savefig(result_png_fp)
        print(f"Result is saved at {result_png_fp}")
        plt.show()
