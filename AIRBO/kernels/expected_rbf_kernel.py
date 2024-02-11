from gpytorch.kernels.kernel import Kernel
from typing import Optional, Tuple
from gpytorch.priors import Prior
from gpytorch.constraints import Interval
import torch


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


def default_postprocess_script(x):
    return x


class ExpectedRBFKernel(Kernel):
    """
    Expected RBF kernel with close-form integral solution
    """
    has_lengthscale = True

    def __init__(self, ard_num_dims: Optional[int] = None,
                 batch_shape: Optional[torch.Size] = torch.Size([]),
                 active_dims: Optional[Tuple[int, ...]] = None,
                 lengthscale_prior: Optional[Prior] = None,
                 lengthscale_constraint: Optional[Interval] = None,
                 eps: Optional[float] = 1e-6,
                 **kwargs):
        super(ExpectedRBFKernel, self).__init__(
            ard_num_dims, batch_shape, active_dims,
            lengthscale_prior, lengthscale_constraint, eps
        )

    def forward(self, x1, x2, diag=False, **params):
        """
        We assume that x1 and x2 are Gaussian variables
        :param x1: a tensor of M * 2D or B * M * 2D, the first D dimensions are the means and the rests are stds
        :param x2: a tensor of N * 2D or B * M * 2D, the first D dimensions are the means and the rests are stds
        :param diag: whether it only needs to compute the diagonal
        :param params: configuration keywords
        :return: a tensor of M * N
        """
        B = x1.shape[:-2] if x1.ndim >= 3 else None
        M = x1.shape[-2]
        N = x2.shape[-2]
        D = x1.shape[-1] // 2
        dtype = x1.dtype
        device = x1.device

        x1_mean = x1[..., 0:D]
        x1_var = x1[..., D:] ** 2.0
        x2_mean = x2[..., 0:D]
        x2_var = x2[..., D:] ** 2.0

        Var1 = torch.diag_embed(x1_var)  # B * M * D * D
        Var2 = torch.diag_embed(x2_var)  # B * N * D * D

        if self.ard_num_dims is None:
            ls_vec = self.lengthscale.repeat(1, D).view(-1)
        else:
            ls_vec = self.lengthscale.view(-1)
        W = torch.diag(ls_vec ** 2.0)  # lengthscale in (D, D)

        # numerator
        AB = x1_mean.unsqueeze(-2) - x2_mean.unsqueeze(-3)  # AB=A-B: broadcast --> (B, M, N, D)
        VAVB = Var1.unsqueeze(-3) + Var2.unsqueeze(-4)  # VAVB = VarA + VarB: (M, N, D, D)
        Z = W.unsqueeze(0).unsqueeze(0) if B is None \
            else W.unsqueeze(0).unsqueeze(0).unsqueeze(
            0) + VAVB  # Z = W + VarA + VarB: (M, N, D, D)
        Z_inv = torch.inverse(Z)  # (M, N, D, D)

        ABZ_eq = 'mnpd,mndd->mnpd' if B is None else 'bmnpd,bmndd->bmnpd'
        ABZ = torch.einsum(ABZ_eq, [AB.unsqueeze(-2), Z_inv])  # (A-B)Z^-1, (M, N, 1, D)
        nu_eq = 'mnpd,mndq->mnpq' if B is None else 'bmnpd,bmndq->bmnpq'
        nu = torch.einsum(nu_eq, [ABZ, AB.unsqueeze(-1)]).squeeze(-1).squeeze(
            -1)  # (A-B)Z^-1(A-B)^T:(M, N, 1, 1)

        # denominator
        x1_eq_x2 = torch.equal(x1, x2)
        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-1], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                res = postprocess_rbf(res)
                return res
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1)
                res = res.pow(2)
                res = postprocess_rbf(res)
                return res
        else:
            W_inv = W.inverse().unsqueeze(0).unsqueeze(0)
            if B is not None:
                W_inv = W_inv.unsqueeze(0)
            de_coeff = (W_inv.matmul(VAVB)).det().sqrt()
            if x1_eq_x2:
                # if the x1 == x2, then it needs separate computations for the diagonal
                # and non-diagonal elements, see the paper appendix C for the details.
                zero_diag_identity = (torch.ones((M, M), dtype=dtype).to(device) -
                                      torch.eye(M, M, dtype=dtype).to(device))
                one_diag = torch.eye(M, M, dtype=dtype).to(device)
                if B is not None:
                    zero_diag_identity = zero_diag_identity.unsqueeze(0)
                    one_diag = one_diag.unsqueeze(0)
                de = de_coeff * zero_diag_identity + one_diag
            else:
                de = de_coeff

            covar = nu / de

            return postprocess_rbf(covar)
