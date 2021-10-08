from typing import Optional

import torch
from torch.autograd.functional import vjp
from torch.optim.optimizer import required

from custom_optimizer.comp_opt import CompositionalOptimizer


class SCGD(CompositionalOptimizer):
    r"""Implements Stochastic Compositional Gradient Descent. `https://arxiv.org/pdf/1411.3803.pdf`
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            -> `params` should be comprised of two dictionaries:
                - `dict(params = X)`
                - `dict(params = Y)`  where Y will track g(X)
            THE ORDER MATTERS
        alpha_start: initial learning rate
        alpha_decay: modulation of learning rate
        beta_start:  initial value of coefficients used for computing running averages
        beta_decay: modulation of the coefficients used for computing running average
    """

    def __init__(self, params: dict, alpha_start: float = 1e-3, alpha_decay: float=.75, beta_start: float = .99, beta_decay: float = .5):
        defaults = dict(alpha_start=alpha_start, beta_start=beta_start, alpha_decay=alpha_decay, beta_decay=beta_decay)

        assert len(params) == 2, "params should be comprised of three dicts (dict(params = X), " \
                                 "                                           dict(params = Y))" \
                                 f"got {len(params)} elements"
        super(SCGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, oracle_g=required, proj_X=required, filter_inds: Optional[torch.Tensor] = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            oracle_g (callable): Function g that will take `x` as input
            proj_X (callable): Function that computes orthogonal projection onto X (inplace modification)
                    filter_inds (callable): Filter of the indices to use for `y`
        Returns:
            g_x (Tensor, optional): result of evaluation g(x)  (which can be used to get actual loss f(g(x)))
        """

        # there should be two param_groups one for X, Y in that ORDER
        assert len(self.param_groups) == 2
        param_group_X, param_group_Y = self.param_groups
        assert len(param_group_X['params']) == len(param_group_Y['params']) == 1

        for i in range(len(param_group_X['params'])):
            p_x, p_y = param_group_X['params'][i], param_group_Y['params'][i]

            if p_y.grad is None:
                raise RuntimeError('param y should have grad as its associated param x does')

            grad_f_y = p_y.grad.double()  # should contain ∇f(y)

            if grad_f_y.is_sparse:
                raise RuntimeError('CAdam does not support sparse gradients, please consider _ instead')
            if filter_inds is not None:
                grad_f_y = grad_f_y[..., filter_inds]

            state_x = self.state[p_x]

            # State initialization
            if len(state_x) == 0:
                state_x['step'] = 0

            state_x['step'] += 1
            step = state_x['step']

            # scheduling
            alpha_decay = param_group_X['alpha_decay']
            alpha_t = param_group_X['alpha_start'] / step ** alpha_decay
            beta_decay = param_group_X['beta_decay']
            beta_t = param_group_X['beta_start'] / step ** beta_decay

            # compute approx of ∇F as ∇g^T(x) ∇f(y) (where F(x) = f(g(x)))
            with torch.enable_grad():
                g_x, grad_F_x = vjp(oracle_g, p_x, grad_f_y, strict=True)

            # Update `x` with gradient step (Algo 1 - step 5:)
            proj_X(p_x.add_(grad_F_x, alpha= - alpha_t))

            # update `y` tracking g(x) (Algo 1 - step 2:)
            p_y.mul_(1 - beta_t)
            if filter_inds is None:
                p_y.add_(oracle_g(p_x), alpha=beta_t)
            else:
                p_y[..., filter_inds].add_(oracle_g(p_x), alpha=beta_t)
        return g_x
