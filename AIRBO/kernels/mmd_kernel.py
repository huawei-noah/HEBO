"""
Maximum Mean Discrepancy (MMD) Kernel
"""
import gpytorch as gpyt
from gpytorch.kernels.kernel import Kernel
import torch
from tqdm.auto import trange
import gc
from utils import commons as cm
from functools import partial


def additive_RQ_kernel(alphas=(0.2, 0.5, 1, 2, 5), ls=1.0, learnable_ls=False):
    assert len(alphas) > 0
    _k_list = []
    for a in alphas:
        _k = gpyt.kernels.RQKernel()
        _k.alpha = a
        _k.lengthscale = ls
        _k.raw_lengthscale.require_grad = learnable_ls
        _k_list.append(_k)
    k = gpyt.kernels.AdditiveKernel(*_k_list)
    return k


def combo_kernel(alphas=(0.2, 0.5, 1, 2, 5), ls=1.0, learnable_ls=False):
    assert len(alphas) > 0
    _k_list = []
    for a in alphas:
        _k = gpyt.kernels.RQKernel()
        _k.alpha = a
        _k.lengthscale = ls
        _k.raw_lengthscale.require_grad = learnable_ls
        _k_list.append(_k)
    _k_list.append(gpyt.kernels.LinearKernel())
    k = gpyt.kernels.AdditiveKernel(*_k_list)
    return k


def postprocess_mmd(dist_mat):
    sym_dist_mat = (dist_mat + dist_mat.T) * 0.5
    _dist_mat = sym_dist_mat.clamp_(min=0) ** 0.5
    return torch.clamp(1.0 - _dist_mat, 0.0, 1.0)


def postprocess_mmd_rbf(dist_mat):
    _dist_mat = ((dist_mat + dist_mat.transpose(-2, -1)) * 0.5)
    return _dist_mat.div_(-2).exp_()


def nystrom_mmd(X, Y, kernel: Kernel, sub_samp_size: int = 100):
    """
    nystrom estimator for MMD2
    :param X: B * D tensor
    :param Y: H * D tensor
    :param kernel: gpytorch.kernel.Kernel instance
    :param sub_samp_size: the subsample number
    :return: a scalar
    """
    # sub-sampling
    B, D = X.shape
    x_sub_inds = torch.randperm(B)[:sub_samp_size]
    X_sub = X[x_sub_inds, :]

    H, D = Y.shape
    y_sub_inds = torch.randperm(H)[:sub_samp_size]
    Y_sub = Y[y_sub_inds, :]

    # compute alpha_x
    k_m_x = kernel(X_sub, X_sub).evaluate()
    k_m_x_inv = torch.linalg.pinv(k_m_x)
    k_mn_x = kernel(X_sub, X).evaluate()
    alpha_x = (k_m_x_inv @ k_mn_x @ torch.ones(X.shape[0], 1).type(X.dtype).to(X.device)) \
              / X.shape[0]

    # compute alpha_y
    k_m_y = kernel(Y_sub, Y_sub).evaluate()
    k_m_y_inv = torch.linalg.pinv(k_m_y)
    k_mn_y = kernel(Y_sub, Y).evaluate()
    alpha_y = (k_m_y_inv @ k_mn_y @ torch.ones(Y.shape[0], 1).type(Y.dtype).to(Y.device)) \
              / Y.shape[0]

    # nystrom estimator
    part1 = alpha_x.T @ k_m_x @ alpha_x
    part2 = alpha_y.T @ k_m_y @ alpha_y
    part3 = alpha_x.T @ kernel(X_sub, Y_sub).evaluate() @ alpha_y * -2

    mmd2 = part1 + part2 + part3

    return mmd2


def nystrom_mmd_batch(X, Y, kernel: Kernel, sub_samp_size: int = 100):
    """
    nystrom estimator for MMD2 in batches
    :param X: B * M * C * D tensor, B is the batch size, M is the sample size,
              C is the sampling size, and D is the data dim.
    :param Y: B * N * H * D tensor, B is the batch size, N is the sample size,
              H is the sampling size, and D is the data dim.
    :param kernel: A gpytorch.kernel.Kernel instance
    :param sub_samp_size: the subsampling size
    :return: a tensor of M * N
    """
    km_x_inv = None
    kmn_x = None
    ones_x = None
    km_y_inv = None
    kmn_y = None
    ones_y = None
    km_x, km_y = None, None
    alpha_x, km_xy = None, None
    alpha_y, X_sub, Y_sub = None, None, None
    part1, part2, part3 = None, None, None
    try:
        B = X.shape[:-3]
        M, C, D = X.shape[-3:]
        N, H, D = Y.shape[-3:]
        _original_shape = kernel.batch_shape

        # sub-sampling
        x_sub_inds = torch.randperm(C)[:sub_samp_size]
        X_sub = X[..., x_sub_inds, :]  # M * m * D, m is the sub-sampling size

        y_sub_inds = torch.randperm(H)[:sub_samp_size]
        Y_sub = Y[..., y_sub_inds, :]  # N * m * D

        # compute alpha for x variables
        kernel.batch_shape = torch.Size([M])
        km_x = kernel(X_sub, X_sub).evaluate()  # B * M * m * m
        km_x_inv = torch.linalg.pinv(km_x)  # B * M * m * m
        kmn_x = kernel(X_sub, X).evaluate()  # B * M * m * C
        ones_x = torch.ones(M, C, 1).type(X.dtype).to(X.device)  # M * C * 1
        alpha_x = (km_x_inv @ kmn_x @ ones_x) / C  # B * M * m * 1
        # cm.free_memory([km_x_inv, kmn_x, ones_x], debug=False)

        # compute alpha for y variables
        kernel.batch_shape = torch.Size([N])
        km_y = kernel(Y_sub, Y_sub).evaluate()  # B * N * m * m
        km_y_inv = torch.linalg.pinv(km_y)  # B * N * m * m
        kmn_y = kernel(Y_sub, Y).evaluate()  # B * N * m * H
        ones_y = torch.ones(N, H, 1).type(Y.dtype).to(Y.device)  # M * H * 1
        alpha_y = (km_y_inv @ kmn_y @ ones_y) / H  # B * N * m * 1
        # cm.free_memory([km_y_inv, kmn_y, ones_y], debug=False)

        # nystrom estimator
        part1 = (alpha_x.transpose(-2, -1) @ km_x @ alpha_x).view(*B, M, 1)  # a_x^T * km_x * a_x
        part2 = (alpha_y.transpose(-2, -1) @ km_y @ alpha_y).view(*B, 1, N)  # a_y^T * km_y * a_y
        # cm.free_memory([km_x, km_y], debug=False)

        kernel.batch_shape = torch.Size([M, N])
        km_xy = kernel(X_sub.unsqueeze(-3),
                       Y_sub.unsqueeze(-4)).evaluate()  # broadcast to M * N --> M * N * m * m
        # a_x^T * km_xy * a_y
        part3 = (alpha_x.unsqueeze(-3).transpose(-2, -1) @ km_xy @ alpha_y.unsqueeze(-4)).view(*B,
                                                                                               M, N)

        mmd2 = part1 + part2 - part3 * 2.0  # broadcast to M * n and add
        # cm.free_memory([alpha_x, km_xy, alpha_y, X_sub, Y_sub, part1, part2, part3], debug=False)
        kernel.batch_shape = _original_shape
    finally:
        cm.free_memory(
            [
                km_x_inv, kmn_x, ones_x,
                km_y_inv, kmn_y, ones_y,
                km_x, km_y,
                alpha_x, km_xy, alpha_y, X_sub, Y_sub, part1, part2, part3,
            ],
            debug=False
        )

    return mmd2


def empirical_mmd(X, Y, kernel: Kernel):
    """
    estimate the MMD
    :param X: B*D tensor
    :param Y: H*D tensor
    :param kernel: gpytorch.kernels.Kernel
    :return: a scalar
    """
    # xx
    cm_xx = kernel(X, X).evaluate()
    avg_xx_mmd = (cm_xx.sum() - torch.diagonal(cm_xx).sum()) / (X.shape[0] * (X.shape[0] - 1))

    # yy
    cm_yy = kernel(Y, Y).evaluate()
    avg_yy_mmd = (cm_yy.sum() - torch.diagonal(cm_yy).sum()) / (Y.shape[0] * (Y.shape[0] - 1))

    # xy
    cm_xy = kernel(X, Y).evaluate()
    avg_xy_mmd = cm_xy.sum() / (X.shape[0] * Y.shape[0])

    mmd = avg_xx_mmd + avg_yy_mmd - 2.0 * avg_xy_mmd

    return mmd


def estimate_mmd_in_chunks(X, Y, estimator, chunk_size=10):
    """
    estimate the MMD in chunks, decrease the chunk size if GPU memory error happens
    :param X: B * M * C * D or M * C * D tensor
    :param Y: B * N * H * D or N * H * D tensor
    :param estimator:
    :param chunk_size:
    :return:
    """
    N = Y.shape[-3]

    retry_i = 0
    success = False
    _practicable_chunk_size = chunk_size
    while not success:
        try:
            _practicable_chunk_size = max(chunk_size // (2 ** retry_i), 1)
            mmd_results = []
            for bs in range(0, N, _practicable_chunk_size):
                be = min(N, bs + _practicable_chunk_size)
                _Y = Y[..., bs:be, :, :]
                _mmd = estimator(X, _Y)
                mmd_results.append(_mmd)

                gc.collect()
                torch.cuda.empty_cache()
            success = True
        except RuntimeError as e:
            if 'CUDA out of memory' in e.args[0] or 'not enough memory' in e.args[0]:
                if _practicable_chunk_size > 1:  # we can still try to reduce the chunk size
                    print(f'Chunk size {_practicable_chunk_size} is too large, '
                          f'reduce it by a half:', e)
                    retry_i += 1

                    if len(mmd_results) > 0:
                        cm.free_memory(mmd_results)
                        for _m in mmd_results:
                            del _m
                        del mmd_results
                    if '_mmd' in locals():
                        cm.free_memory([_mmd])
                    if '_Y' in locals():
                        cm.free_memory([_Y])
                else:
                    raise ValueError('Chunk size has been reduced to 1 but still out of memory, '
                                     'try cpu.')
            else:
                raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    mmd = torch.concat(mmd_results, dim=-1)  # concat to columns as we chuck the Y
    return mmd


def empirical_mmd_batch(X, Y, kernel: gpyt.kernels.Kernel):
    """
    Empirically estimate the MMD in batches
    :param X: M * B * D tensor, M is the batch size, B is the sample size, D is the data dimension
    :param Y: N * H * D tensor, N is the batch size, H is the sample size, D is the data dimension
    :param kernel: the kernel to use
    :return: a tensor of M*N
    """
    B = X.shape[:-3]
    M, C, D = X.shape[-3:]
    N, H, D = Y.shape[-3:]
    _original_shape = kernel.batch_shape

    cm_xx = None
    cm_yy = None
    cm_xy = None
    try:

        # compute the covariance btw X and X
        kernel.batch_shape = torch.Size([M])  # align along M, and compute kernel for each pair
        cm_xx = kernel(X, X).evaluate()  # M * B * B
        avg_xx_mmd = (cm_xx.sum((-2, -1)) - torch.diagonal(cm_xx, dim1=-2, dim2=-1).sum(-1)) \
                     / (C * (C - 1))  # M * 1

        # compute the covariance btw Y and Y
        kernel.batch_shape = torch.Size([N])  # align along N, and compute kernel for each pair
        cm_yy = kernel(Y, Y).evaluate()  # N x H x H
        avg_yy_mmd = (cm_yy.sum((-2, -1)) - torch.diagonal(cm_yy, dim1=-2, dim2=-1).sum(-1)) \
                     / (H * (H - 1))

        # compute the covariance btw X and Y
        kernel.batch_shape = torch.Size([M, N])  # make a grid of M * N, apply kernel on each pair
        cm_xy = kernel(X.unsqueeze(-3), Y.unsqueeze(-4)).evaluate()  # broadcast to M * N, output M x N x B x H
        avg_xy_mmd = cm_xy.sum((-2, -1)) / (C * H)

        mmd2 = avg_xx_mmd.unsqueeze(-1) + avg_yy_mmd.unsqueeze(-2) - 2.0 * avg_xy_mmd  # broadcast and elementwise add

    finally:
        kernel.batch_shape = _original_shape
        cm.free_memory(
            [cm_xx, cm_yy, cm_xy, ],
            debug=False
        )
    return mmd2


class MMDKernel(Kernel):
    """
    Decorating an existing kernel with MMD distance
    """

    has_lengthscale = True

    def __init__(self, base_kernel, **kwargs):
        super(MMDKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.chunk_size = kwargs.get('chunk_size', 100)
        self.estimator = kwargs.get('estimator', 'nystrom')
        self.sub_samp_size = kwargs.get('sub_samp_size', 100)
        self.estimation_trials = kwargs.get('estimation_trials', 1)
        if self.estimator == 'nystrom':
            self.estimation_func = partial(
                nystrom_mmd_batch, kernel=self.base_kernel, sub_samp_size=self.sub_samp_size
            )
        elif self.estimator == 'empirical':
            self.estimation_func = partial(empirical_mmd_batch, kernel=self.base_kernel)
        else:
            raise ValueError('Unsupported estimator name', self.estimator)

    # @property
    # def lengthscale(self):
    #     ls = None
    #     if isinstance(self.base_kernel, gpyt.kernels.AdditiveKernel):
    #         ls = torch.concat([_k.lengthscale for _k in self.base_kernel.kernels], dim=0)
    #     else:
    #         ls = self.base_kernel.lengthscale
    #     return ls

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def compute_distance_covariance_matrix(self, x1, x2, diag=False, **params):
        chunk_size = params.get('chunk_size', self.chunk_size)
        avg_dist_mat = None
        for _ in range(self.estimation_trials):
            dist_mat = estimate_mmd_in_chunks(x1, x2, self.estimation_func, chunk_size)
            avg_dist_mat = dist_mat if avg_dist_mat is None else avg_dist_mat + dist_mat
        avg_dist_mat = avg_dist_mat / self.estimation_trials
        cov_mat = postprocess_mmd_rbf(avg_dist_mat.div(self.lengthscale ** 2.0))
        return avg_dist_mat, cov_mat

    def forward(self, x1, x2, diag=False, **params):
        avg_dist_mat, cov_mat = self.compute_distance_covariance_matrix(x1, x2, diag, **params)
        return cov_mat


# %%
if __name__ == '__main__':
    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from itertools import product
    from torch import distributions
    from functools import partial
    import seaborn as sns
    from tqdm.auto import trange, tqdm
    from utils import input_uncertainty as iu
    from scipy import stats

    # %%
    # general setup
    n_vars = 1
    dtype = torch.float
    device = torch.device('cpu')
    chunk_size = 512

    # %%
    # # test the batch implementation
    # n_trials = 500
    # n_samp = 500
    # X = distributions.Normal(0.0, 1.0).sample((1, n_samp, 1)).type(dtype).to(device)
    # Y = distributions.Normal(1.0, 1.0).sample((n_trials, n_samp, 1)).type(dtype).to(device)
    # k = gpyt.kernels.LinearKernel().to(device)
    # with torch.no_grad():
    #     b_emp = empirical_mmd_batch(X, Y, k).detach().cpu().numpy().flatten()
    #     s_emp = [empirical_mmd(X[0], Y[y_i], k).item() for y_i in range(n_trials)]
    #     b_nyst = nystrom_mmd_batch(X, Y, k, 100).detach().cpu().numpy().flatten()
    #     s_nyst = [nystrom_mmd(X[0], Y[y_i], k, 100).item() for y_i in range(n_trials)]
    #
    # # visualize
    # nrows, ncols, ax_scale = 2, 2, 1.0
    # fig = plt.figure(figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    # axes = fig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False).flatten()
    # bins = np.arange(0.4, 1.2, 0.01)
    # cmap = plt.get_cmap('tab10')
    # for ax_i, (est_name, est_val) in enumerate(
    #         {'batch_emp': b_emp, 'sample_emp': s_emp,
    #          'batch_nyst': b_nyst, 'sample_nyst': s_nyst}.items()
    # ):
    #     axes[ax_i].hist(
    #         est_val, bins=bins,
    #         label=f'[{est_name}] $\mu={np.mean(est_val):.3f}, \sigma={np.std(est_val): .3f}$',
    #         alpha=0.7, edgecolor='k', facecolor=cmap.colors[ax_i]
    #     )
    #     axes[ax_i].legend()
    # fig.tight_layout()
    # plt.show()

    # %%
    # # test the nystrom estimator
    # x_mean, x_std = 0.0, 1.0
    # y_mean, y_std = 1.0, 1.0
    # n_trials = 400
    # k = gpyt.kernels.LinearKernel().to(device)
    #
    # results = []
    # emp_estimator = partial(empirical_mmd_batch, kernel=k)
    # for samp_size in [50, 100, 500, 1000]:
    #     X = distributions.Normal(x_mean, x_std).sample((n_trials, samp_size, n_vars)) \
    #         .type(dtype).to(device)
    #     Y = distributions.Normal(y_mean, y_std).sample((n_trials, samp_size, n_vars)) \
    #         .type(dtype).to(device)
    #
    #     # empirical estimator
    #     emp_chunk_size = int(
    #         n_trials / np.log10(samp_size) / 2 ** (np.log10(samp_size) + 2.5)
    #     )
    #     print(f'[empirical] #samp={samp_size}, chunk={emp_chunk_size}')
    #     with torch.no_grad():
    #         emp = estimate_mmd_in_chunks(
    #             X, Y, emp_estimator, chunk_size=emp_chunk_size
    #         ).detach().cpu().numpy().flatten()
    #
    #     # nystrom estimator
    #     for sub_samp_ratio in [0.1, 0.5, 0.9]:
    #         sub_samp_size = int(samp_size * sub_samp_ratio)
    #         nyst_estimator = partial(nystrom_mmd_batch, kernel=k, sub_samp_size=sub_samp_size)
    #         nyst_chunk_size = int(
    #             n_trials / np.log10(sub_samp_size) / 2 ** (np.log10(sub_samp_size) + 2.5)
    #         )
    #         print(f'[nystrom] sub-sample:{sub_samp_size}/{samp_size}, chunk: {nyst_chunk_size}')
    #         with torch.no_grad():
    #             nyst = estimate_mmd_in_chunks(
    #                 X, Y, nyst_estimator, nyst_chunk_size
    #             ).detach().cpu().numpy().flatten()
    #         results.append((samp_size, sub_samp_size, emp, nyst))
    #
    # # visualize
    # print('visualizing...')
    # nrows, ncols, ax_scale = int(np.ceil(len(results) / 3)), 3, 1.2
    # fig = plt.figure(figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    # axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharex=False,
    #                     sharey=True).flatten()
    # bins = np.arange(-0.0, 1.5, 0.01)
    # for ax_i, (_samp_size, _sub_size, _emp, _nyst) in enumerate(results):
    #     if _emp is not None:
    #         axes[ax_i].hist(
    #             _emp, bins=bins, alpha=0.5,
    #             label=f'[empirical] $\mu={np.mean(_emp):.3f}, \sigma={np.std(_emp):.3f}$'
    #         )
    #
    #     if _nyst is not None:
    #         axes[ax_i].hist(
    #             _nyst, bins=bins, alpha=0.5,
    #             label=f'[nystrom] $\mu={np.mean(_nyst):.3f}, \sigma={np.std(_nyst):.3f}$'
    #         )
    #     axes[ax_i].legend()
    #     axes[ax_i].set_title(f'#samp={_samp_size}, #sub-samp={_sub_size}')
    # plt.tight_layout()
    # plt.show()

    # %%
    #  examine how the kernel, mean, std and sampling size affect the MMD distance
    y_means = [i for i in np.arange(-1, 1, 0.01)]
    stds = [0.7, ]
    ls_vals = [1.0, ]
    sampling_sizes = [100, ]
    plot_cov = True
    kernels = {
        # 'Linear': gpyt.kernels.LinearKernel(),
        # 'RBF': gpyt.kernels.RBFKernel(),
        # 'RQ': gpyt.kernels.RQKernel(),
        'combo_kernel': combo_kernel([0.2, 0.5, 1, 2, 5]),
        # 'RQS': additive_RQ_kernel([0.2, 0.5, 1, 2, 5]),
        # 'Marten52': gpyt.kernels.MaternKernel(),
        # 'Poly3': gpyt.kernels.PolynomialKernel(3),
        # 'Poly6': gpyt.kernels.PolynomialKernel(6),
    }
    # compute the mmd under different setups
    with torch.no_grad():
        _results = {}
        for (kn, k), _std, ls, n_samp in product(kernels.items(), stds, ls_vals, sampling_sizes):
            # input_distrib = iu.ScipyDistributionInput(
            #     stats.beta(a=0.4, b=0.2, scale=_std), name='beta', n_var=1,
            #     # stats.norm(0.0, scale=_std), name='norm', n_var=1
            # )
            input_distrib = iu.GMMInputDistribution(
                n_components=2, mean=np.array([[-0.017], [0.03]]),
                covar=np.array([0.010 ** 2.0, 0.017 ** 2.0]),
                weight=np.array([0.5, 0.5])
            )

            if k.has_lengthscale:
                k.lengthscale = ls
            # estimator = partial(empirical_mmd_batch, kernel=k)
            estimator = partial(nystrom_mmd_batch, kernel=k)
            x = torch.tensor(input_distrib.sample((n_samp, n_vars)), dtype=dtype, device=device)
            ys = [
                (
                    ym,
                    torch.tensor(
                        input_distrib.sample((n_samp, n_vars)) + ym,
                        dtype=dtype, device=device
                    )
                )
                for ym in y_means]
            k = k.to(device)
            mmd2_distances = {
                ym: estimate_mmd_in_chunks(x.unsqueeze(0), y.unsqueeze(0), estimator, chunk_size)
                for ym, y in ys
            }
            _results[(kn, _std, ls, n_samp)] = mmd2_distances

    # plot
    nrows = 2 if plot_cov else 1
    ncols = len(kernels)
    ax_scale = 1.0
    fig = plt.figure(figsize=(4 * ax_scale * ncols, 3 * ax_scale * nrows))
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharex=True)
    for ax_i, kn in enumerate(kernels.keys()):
        # mmd
        for _std, _ls, _samp in product(stds, ls_vals, sampling_sizes):
            k_distances = _results[(kn, _std, _ls, _samp)]
            axes[0, ax_i].plot(
                [m for m, d in k_distances.items()],
                [d.item() for m, d in k_distances.items()],
                label=f'sampling_size={_samp}', alpha=0.6)
        axes[0, ax_i].set_ylabel('MMD2 estimation')
        # axes[0, ax_i].set_xlabel('Sample mean distance')
        axes[0, ax_i].legend()
        # axes[ax_i, 0].set_title(kn)
        axes[ax_i, 0].set_title('$Beta(0.4, 0.2)$')

        # cov
        if plot_cov:
            for _std, _ls, _samp in product(stds, ls_vals, sampling_sizes):
                k_distances = _results[(kn, _std, _ls, _samp)]
                axes[1, ax_i].plot(
                    [m for m, d in k_distances.items()],
                    [postprocess_mmd_rbf(d).item() for m, d in k_distances.items()],
                    label=f'sampling_size={_samp}', alpha=0.6)
            axes[1, ax_i].set_ylabel('Covariance')
            axes[1, ax_i].set_xlabel('Sample mean distance')
            axes[1, ax_i].legend()
            # axes[1, ax_i].set_title(kn)
    plt.tight_layout()
    plt.show()

    # %%
    # MMD estimation errors
    # n_vars = 1
    # n_trials = 200
    # mmd_estimations = []
    # samp_sizes = [20, 100]
    # k = additive_RQ_kernel().to(device)
    # estimator = partial(empirical_mmd_batch, kernel=k)
    # input_distrib = iu.ScipyDistributionInput(
    #     stats.beta(a=0.4, b=0.2, scale=0.07), name='beta', n_var=1
    # )
    # for n_samp in samp_sizes:
    #     x_samples = torch.tensor(input_distrib.sample((n_trials, n_samp, n_vars)),
    #                              dtype=dtype, device=device)
    #     y_samples = torch.tensor(input_distrib.sample((n_trials, n_samp, n_vars)),
    #                              dtype=dtype, device=device)
    #     z_samples = torch.tensor(input_distrib.sample((n_trials, n_samp, n_vars)),
    #                              dtype=dtype, device=device) + 0.2
    #
    #
    #     with torch.no_grad() and gpyt.settings.lazily_evaluate_kernels(False):
    #         chunk_size = 10 if n_samp >= 200 else 100
    #         xy_mmd_vals = [
    #             postprocess_mmd_rbf(_mmd.view(1, 1)).item()
    #             for _mmd in estimate_mmd_in_chunks(x_samples, y_samples,
    #                                                estimator, chunk_size).view(-1)
    #         ]
    #
    #         xz_mmd_vals = [
    #             postprocess_mmd_rbf(_mmd.view(1, 1)).item()
    #             for _mmd in estimate_mmd_in_chunks(x_samples, z_samples,
    #                                                estimator, chunk_size).view(-1)
    #         ]
    #
    #         mmd_estimations.append((n_samp, xy_mmd_vals, xz_mmd_vals))
    #
    # # visualize
    # nrows, ncols, ax_scale = len(mmd_estimations) // 2, 2, 1.0
    # fig = plt.figure(figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    # axes = fig.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, squeeze=False).flatten()
    # for plot_i, (n_samp, xy_mmd, xz_mmd) in enumerate(mmd_estimations):
    #     axes[plot_i].hist(
    #         xy_mmd, bins=np.arange(0.85, 1.05, 0.005),
    #         label=f'[x=y] $\mu={np.mean(xy_mmd):.3f}, \sigma={np.std(xy_mmd):.3f}$',
    #         alpha=0.7
    #     )
    #     axes[plot_i].hist(
    #         xz_mmd, bins=np.arange(0.85, 1.05, 0.005),
    #         label=f'[x!=z] $\mu={np.mean(xz_mmd):.3f}, \sigma={np.std(xz_mmd):.3f}$',
    #         alpha=0.7
    #     )
    #     axes[plot_i].set_xlabel('mmd covariance')
    #     axes[plot_i].set_ylabel('count')
    #     axes[plot_i].legend()
    #     axes[plot_i].set_title(f'#samp={n_samp}')
    # fig.tight_layout()
    # plt.show()

    # %%
    # MMD for unsymmetrical distribution
    from itertools import product

    # n_vars = 1
    # n_trials = 50
    # mmd_estimations = []
    # samp_size = 1000
    # sub_samp_size = 100
    # k = additive_RQ_kernel().to(device)
    # nyst_estimator = partial(nystrom_mmd_batch, kernel=k, sub_samp_size=sub_samp_size)
    #
    # results = []
    # z_means = range(0, 9)
    # x_mean, x_df = 4.0, 0.5
    # x_samples = torch.distributions.Chi2(x_df).sample((n_trials, samp_size, n_vars)) \
    #                 .type(dtype).to(device) + x_mean
    # for z_mean in tqdm(z_means, desc='evaluating mmd....'):
    #     df = x_df if z_mean <= x_mean else x_df + 3.5
    #     z_df = df
    #     z_samples = torch.distributions.Chi2(z_df).sample((n_trials, samp_size, n_vars)) \
    #                     .type(dtype).to(device) + z_mean
    #     mmd2_vals = estimate_mmd_in_chunks(x_samples, z_samples, nyst_estimator, n_trials)
    #     results.append((df, x_samples, z_samples, z_mean, mmd2_vals))
    #
    # # visualize
    # nrows, ncols, ax_scale = 1, 2, 1.5
    # fig = plt.figure(figsize=(4 * ncols * ax_scale, 3 * nrows * ax_scale))
    # axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    # axes[0].plot([_zm for _df, _x, _z, _zm, _mvs in results],
    #              [_mvs.mean().item() for _df, _x, _z, _zm, _mvs in results],
    #              ls='--', c='tab:grey', marker='.')
    # for _df, _x, _z, _zm, _mvs in results:
    #     axes[0].errorbar(
    #         [_zm],
    #         y=[_mvs.mean().item()],
    #         yerr=[_mvs.std().item() * 2.0],
    #         label=f'z_df={_df}'
    #     )
    # axes[0].legend()
    # axes[0].set_ylabel('mmd2')
    # axes[0].set_xlabel('z_mean')
    # axes[0].set_title("MMD2 under different z_mean")
    #
    # sns.kdeplot(x_samples.detach().cpu().numpy().flatten(), ax=axes[1],
    #             label=f'x~$\chi^2$({x_mean}, {x_df:.2f})', lw=2, ls='--')
    # for _df, _x, _z, _zm, _mvs in results:
    #     sns.kdeplot(_z.detach().cpu().numpy().flatten(), ax=axes[1],
    #                 label=f'z~$\chi^2$({_zm}, {_df:.1f})')
    # axes[1].set_xlim(-1, 10)
    # axes[1].legend()
    # axes[1].set_title("x distribution under different df")
    #
    # fig.tight_layout()
    # plt.show()

    # %%
    # # prepare data
    # n_vars = 1
    # x_means = [i for i in range(100)]
    # _std = 2.0
    # x_stds = [_std, ] * len(x_means)
    #
    # n_samp_size = 100
    # X = []
    # for m, s in zip(x_means, x_stds):
    #     distrib = torch.distributions.Normal(m, s)
    #     X.append(distrib.sample((n_samp_size, n_vars)))
    # X = torch.stack(X)
    #
    # # %%
    # # compute the covariance matrix
    # print(f'X.shape={X.shape}')
    # inner_kernel = gpyt.kernels.LinearKernel()
    # k = MMDKernel(inner_kernel=inner_kernel)
    # with torch.no_grad() and gpyt.settings.lazily_evaluate_kernels(False):
    #     dm_XX, cm_XX = k.forward_ex(X, X)
    #     dm_XX = dm_XX.detach().cpu().numpy()
    #     cm_XX = cm_XX.detach().cpu().numpy()
    # print(f'cm_XX.shape={cm_XX.shape}')
    # # %%
    # # plot
    # cm = cm_XX
    # dm = dm_XX
    # x_means = X.squeeze().mean(dim=1).detach().cpu().numpy()
    #
    # plot_dm = True
    # nrows = 2 if plot_dm else 1
    # ncols = 1
    # fig_scale = 4
    # fig = plt.figure(figsize=(4 * fig_scale * ncols, 3 * fig_scale * nrows))
    # axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    #
    # cmap = plt.get_cmap('Spectral_r')
    # mat_list = [('cm', cm)]
    # if plot_dm:
    #     mat_list.append(('dm', dm))
    # for ax_i, (mat_name, mat) in enumerate(mat_list):
    #     sc = axes[ax_i].matshow(mat, cmap=cmap)
    #     for i in range(X.shape[0]):
    #         for j in range(X.shape[0]):
    #             text = axes[ax_i].text(j, i, f'{mat[i, j]:.2f}',
    #                                    ha="center", va="center", color="w", fontsize=14)
    #     axes[ax_i].set_xticks(range(X.shape[0]), [f'{xm:.2f}' for xm in x_means], rotation=90)
    #     axes[ax_i].set_yticks(range(X.shape[0]), [f'{xm:.2f}' for xm in x_means], rotation=0)
    #     axes[ax_i].set_title(mat_name)
    #     plt.colorbar(sc, ax=axes[ax_i])
    # fig.suptitle(f'std={_std:.2f}')
    # fig.tight_layout()
    # plt.show()
