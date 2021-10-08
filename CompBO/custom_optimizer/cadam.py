import math
from typing import Optional, Callable

import torch
from torch.autograd.functional import vjp
from torch.optim.optimizer import required

from custom_optimizer.comp_opt import CompositionalOptimizer


class CAdam(CompositionalOptimizer):
    r"""Implements Compositional Adam algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
            -> in this first version of CAdam, `params` should be comprised of two dictionaries:
                - `dict(params = X)`
                - `dict(params = Y)`  where Y will track g(X)
            THE ORDER MATTERS
        lr (float, optional): learning rate (default: 1e-3), at step `t` actual learning rate is `lr` / t**(1 / 5)
        beta (float): coefficient used for computing updates of `y_t` and `z_t (see CAdam algorithm)
        mu (float): auxilliary coefficient used for computing running averages of gradient and its square
        C_gamma (float): coefficient used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, beta=0.01, mu=0.9, C_gamma=1, alpha_decay: float = 0.2, mu_decay: float = 1,
                 gamma2_decay: float = 0.4, eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:  # corresponds to C_alpha in CAdam
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < beta <= 1.0:  # corresponds to C_beta in CAdam
            raise ValueError("Invalid beta value: {}".format(beta))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 < mu < 1.0:
            raise ValueError("Invalid gamma parameter at index 0: {}".format(mu))
        if not 0.0 <= C_gamma <= 1.0:
            raise ValueError("Invalid gamma parameter at index 1: {}".format(C_gamma))
        if not alpha_decay > 0:
            raise ValueError("Invalid `alpha_decay` parameter: {}".format(alpha_decay))
        if not mu_decay > 0:
            raise ValueError("Invalid `mu_decay` parameter: {}".format(mu_decay))
        if not gamma2_decay > 0:
            raise ValueError("Invalid `gamma2_decay` parameter: {}".format(gamma2_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, beta=beta, mu=mu, eps=eps, C_gamma=C_gamma, alpha_decay=alpha_decay, mu_decay=mu_decay,
                        gamma2_decay=gamma2_decay, weight_decay=weight_decay)

        assert len(params) == 2, "params should be comprised of three dicts (dict(params = X), " \
                                 "                                           dict(params = Y))" \
                                 f"got {len(params)} elements"
        super(CAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None, oracle_g=required, oracle_y_g: Optional[Callable] = None,
             filter_inds: Optional[torch.Tensor] = None, filter_inds_y_update: Optional[torch.Tensor] = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            oracle_g (callable): Function g that will be used for the computation of the gradient of g(x)
            filter_inds (callable): Filter of the indices to use for y
            oracle_y_g: `g` oracle used to update `y` with `g(z)`
            filter_inds_y_update:  Filter of the indices to use for y update
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
                raise RuntimeError('CAdam does not support sparse gradients, please consider _ instead')

            state_x = self.state[p_x]

            # State initialization
            if len(state_x) == 0:
                state_x['step'] = 0
                # Exponential moving average of gradient values
                state_x['exp_avg'] = torch.zeros_like(p_x, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state_x['exp_avg_sq'] = torch.zeros_like(p_x, memory_format=torch.preserve_format)

            state_x['step'] += 1
            step = state_x['step']

            # scheduling
            mu_t = param_group_X['mu'] ** (param_group_X['mu_decay'] * step)
            gamma1_t = param_group_X['C_gamma'] * mu_t
            gamma2_t = 1 - param_group_X['lr'] / step ** (param_group_X['gamma2_decay']) * (
                    1 - param_group_X['C_gamma'] * mu_t)**2
            beta_t = param_group_X['beta']

            eps = param_group_X['eps']

            # z update (part I)
            p_z = p_x.clone().mul(1 - 1 / beta_t)

            exp_avg_x, exp_avg_sq_x = state_x['exp_avg'], state_x['exp_avg_sq']

            # compute ∇J as ∇g^T(x) ∇f(y) (not exactly ∇J(x))
            with torch.enable_grad():
                g_x, grad_J_x = vjp(oracle_g, p_x, grad_f_y, strict=True)
            # Decay the first and second moment running average coefficient
            exp_avg_x.mul_(gamma1_t).add_(grad_J_x, alpha=1 - gamma1_t)
            exp_avg_sq_x.mul_(gamma2_t).addcmul_(grad_J_x, grad_J_x, value=1 - gamma2_t)

            bias_correction1 = 1 - gamma1_t
            bias_correction2 = 1 - gamma2_t

            # x update
            alpha_t = param_group_X['lr'] / step ** (param_group_X['alpha_decay']) / bias_correction1
            p_x.addcdiv_(exp_avg_x, exp_avg_sq_x.sqrt().add_(eps) / math.sqrt(bias_correction2), value=-alpha_t)

            # z update (part 2)
            p_z.add_(p_x, alpha=1 / beta_t)

            # update y
            if oracle_y_g is None:
                oracle_y_g = oracle_g

            y_up = oracle_y_g(p_z)
            p_y.mul_(1 - beta_t)
            if filter_inds_y_update is None:
                p_y.add_(y_up, alpha=beta_t)
            else:
                p_y[..., filter_inds_y_update] += y_up * beta_t

        return g_x
