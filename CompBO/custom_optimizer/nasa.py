import math
from typing import Optional, Callable

import torch
from torch.autograd.functional import vjp
from torch.optim.optimizer import Optimizer, required

from custom_optimizer.comp_opt import CompositionalOptimizer


class NASA(CompositionalOptimizer):
    r"""Implements Nested Averaged Stochastic Approximation. `https://arxiv.org/pdf/1812.01094.pdf`
    To ease comparison with CAdam, we change notation of the paper to:
        - `u` tracking `g(x)` -> `y`
        - `z` tracking `∇(f ⚬ g))(x)` -> `m`
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            -> in this first version of CAdam, `params` should be comprised of two dictionaries:
                - `dict(params = X)`
                - `dict(params = Y)`  where Y will track g(X) (REMARK: in the original article, it was called `u`)
            THE ORDER MATTERS
        a (float): auxilliary coefficient to define learning rates
        b (float): auxilliary coefficient to define learning rates
        beta: regularization variable
        gamma: decay rate
    """

    def __init__(self, params, a: float = 1., b: float = 1., beta: float = 1, gamma=.6):
        defaults = dict(a=a, b=b, beta=beta, gamma=gamma)

        assert len(params) == 2, "params should be comprised of three dicts (dict(params = X), " \
                                 "                                           dict(params = Y))" \
                                 f"got {len(params)} elements"
        super(NASA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, oracle_g=required, proj_X=required,
             filter_inds: Optional[torch.Tensor] = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            oracle_g (callable): Function g that will take `x` as input
            proj_X (callable): Function that computes orthogonal projection onto X
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

            grad_f_y = p_y.grad.to(p_x)  # should contain ∇f(y)
            if filter_inds is not None:
                grad_f_y = grad_f_y[..., filter_inds]

            if grad_f_y.is_sparse:
                raise RuntimeError('NASA does not support sparse gradients, consider _ instead')

            state_x = self.state[p_x]

            # State initialization
            if len(state_x) == 0:
                state_x['step'] = 0
                # Exponential moving average of gradient values
                state_x['exp_avg'] = torch.zeros_like(p_x, memory_format=torch.preserve_format)

            state_x['step'] += 1
            step = state_x['step']

            # scheduling
            a = param_group_X['a']
            b = param_group_X['b']
            beta_t = param_group_X['beta']
            gamma = param_group_X['gamma']

            tau_t = 1 / (step ** gamma * a)

            exp_avg_x = state_x['exp_avg']  # z^k in the paper
            aux_x_m = proj_X(p_x - 1 / beta_t * exp_avg_x)  # `y^k` in the paper (2.5)

            # Update x (2.6)
            p_x.mul_(1 - tau_t).add_(aux_x_m, alpha=tau_t)

            # compute ∇F as ∇g^T(x) ∇f(y) (not exactly ∇F(x), where F(x) = f(g(x)))
            with torch.enable_grad():
                g_x, grad_F_x = vjp(oracle_g, p_x, grad_f_y, strict=True)

            # Update gradient tracker (2.7)
            exp_avg_x.mul_(1 - a * tau_t).add_(grad_F_x, alpha=a * tau_t)

            # update `y` ((2.8) in the paper -> update of `u`)
            p_y.mul_(1 - b * tau_t)
            if filter_inds is None:
                p_y.add_(g_x, alpha=b * tau_t)
            else:
                p_y[..., filter_inds].add_(g_x, alpha=b * tau_t)
        return g_x
