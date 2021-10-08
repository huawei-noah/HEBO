import math

import torch
from torch.optim.optimizer import Optimizer


class Adamos(Optimizer):
    r"""Implements Adam-like optimization steps with CAdam scheduling.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3), at step `t` actual learning rate is `lr` / t**(1 / 5)
        mu (float): auxilliary coefficient used for computing running averages of gradient and its square
        C_gamma (float): coefficient used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, mu=0.9, C_gamma=1, alpha_decay: float = 0.2, mu_decay: float = 1,
                 gamma2_decay: float = 0.4, eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:  # corresponds to C_alpha in CAdam
            raise ValueError("Invalid learning rate: {}".format(lr))
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
        defaults = dict(lr=lr, mu=mu, eps=eps, C_gamma=C_gamma, alpha_decay=alpha_decay, mu_decay=mu_decay,
                        gamma2_decay=gamma2_decay, weight_decay=weight_decay)

        assert len(params) == 1, "params should be comprised of three dicts (dict(params = X))" \
                                 f"got {len(params)} elements"
        super(Adamos, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adamos, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # there should be one param_group X
        assert len(self.param_groups) == 1
        param_group_X = self.param_groups[0]
        assert len(param_group_X['params']) == 1

        for i in range(len(param_group_X['params'])):
            p_x = param_group_X['params'][i]

            grad_J = p_x.grad.double()  # should contain ∇f(g(x))
            assert grad_J.ndim == 3, f"tensor gradient of f is of rank {grad_J.ndim} should be of rank 3"

            if grad_J.is_sparse:
                raise RuntimeError('Adamos does not support sparse gradients, please consider _ instead')

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

            eps = param_group_X['eps']

            exp_avg_x, exp_avg_sq_x = state_x['exp_avg'], state_x['exp_avg_sq']

            # Decay the first and second moment running average coefficient
            exp_avg_x.mul_(gamma1_t).add_(grad_J, alpha=1 - gamma1_t)
            exp_avg_sq_x.mul_(gamma2_t).addcmul_(grad_J, grad_J, value=1 - gamma2_t)

            bias_correction1 = 1 - gamma1_t
            bias_correction2 = 1 - gamma2_t

            # x update
            alpha_t = param_group_X['lr'] / step ** (param_group_X['alpha_decay']) / bias_correction1
            p_x.addcdiv_(exp_avg_x, exp_avg_sq_x.sqrt().add_(eps) / math.sqrt(bias_correction2), value=-alpha_t)

        return loss
