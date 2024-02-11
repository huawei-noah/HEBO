"""
Kernel-mean-embedding Kernel defined in uGP-UCB:
    K(X, X`) = \int_x \int_x` k(x, x`) dP(x) dP(x`)
"""
from gpytorch.kernels.kernel import Kernel
import gc
from utils import commons as cm
from functools import partial
import torch


def postprocess_linear(dist_mat):
    sym_dist_mat = ((dist_mat + dist_mat.transpose(-2, -1)) * 0.5)
    return sym_dist_mat


def integral_KME_batch(X, Y, kernel: Kernel):
    """
    Estimate the KME via sampling and integral
    :param X: B * M * C * D tensor, B is the batch size, M is the sample size,
              C is the sampling size, and D is the data dim.
    :param Y: B * N * H * D tensor, B is the batch size, N is the sample size,
              H is the sampling size, and D is the data dim.
    :param kernel: A gpytorch.kernel.Kernel instance
    :return: a tensor of M * N
    """
    dist_mat = None
    try:
        B = X.shape[:-3]
        M, C, D = X.shape[-3:]
        N, H, D = Y.shape[-3:]
        _original_shape = kernel.batch_shape
        kernel.batch_shape = torch.Size([M, N])
        dist_mat = kernel(
            X.unsqueeze(-3),  # cast into B * M * 1 * m * D
            Y.unsqueeze(-4)  # cast into B * 1 * N * m * D
        ).evaluate()  # B * M * N * m * m

        kme = dist_mat.mean(dim=[-1, -2])
        kernel.batch_shape = _original_shape
    finally:
        cm.free_memory(
            [
                dist_mat,
            ],
            debug=False
        )

    return kme


def estimate_KME_in_chunks(X, Y, estimator, chunk_size=10):
    """
    Estimate the KME in chunks, decrease the chunk size if GPU memory error happens
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
    kme_results = None
    while not success:
        kme_results = []
        _Y = None
        _kme = None
        try:
            _practicable_chunk_size = max(chunk_size // (2 ** retry_i), 1)
            for bs in range(0, N, _practicable_chunk_size):
                be = min(N, bs + _practicable_chunk_size)
                _Y = Y[..., bs:be, :, :]
                _kme = estimator(X, _Y)
                kme_results.append(_kme)

                gc.collect()
                torch.cuda.empty_cache()
            success = True
        except RuntimeError as e:
            if 'CUDA out of memory' in e.args[0] or 'not enough memory' in e.args[0]:
                if _practicable_chunk_size > 1:  # we can still try to reduce the chunk size
                    print(f'Chunk size {_practicable_chunk_size} is too large, '
                          f'reduce it by a half:', e)
                    retry_i += 1

                    if len(kme_results) > 0:
                        cm.free_memory(kme_results)
                        for _m in kme_results:
                            del _m
                        del kme_results
                    if '_kme' in locals():
                        cm.free_memory([_kme])
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

    kme = torch.concat(kme_results, dim=-1)  # concat to columns as we chuck the Y
    return kme


class KMEKernel(Kernel):
    """
    Decorating an existing kernel with KME
    """

    has_lengthscale = True

    def __init__(self, base_kernel, **kwargs):
        super(KMEKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.chunk_size = kwargs.get('chunk_size', 100)
        self.estimator = kwargs.get('estimator', 'integral')
        self.estimation_trials = kwargs.get('estimation_trials', 1)
        if self.estimator == 'integral':
            self.estimation_func = partial(
                integral_KME_batch, kernel=self.base_kernel
            )
        else:
            raise ValueError('Unsupported estimator name', self.estimator)

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
            dist_mat = estimate_KME_in_chunks(x1, x2, self.estimation_func, chunk_size)
            avg_dist_mat = dist_mat if avg_dist_mat is None else avg_dist_mat + dist_mat
        avg_dist_mat = avg_dist_mat / self.estimation_trials
        cov_mat = postprocess_linear(avg_dist_mat.div(self.lengthscale ** 2.0))
        return avg_dist_mat, cov_mat

    def forward(self, x1, x2, diag=False, **params):
        avg_dist_mat, cov_mat = self.compute_distance_covariance_matrix(x1, x2, diag, **params)
        return cov_mat
