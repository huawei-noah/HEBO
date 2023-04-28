import numpy as np

from febo.algorithms.safety import SafetyMixin
from .acquisition import AcquisitionAlgorithm
from .model import ModelMixin


class UCB(ModelMixin, AcquisitionAlgorithm):
    """
    Implements the Upper Confidence Bound (UCB) algorithm.
    """

    def initialize(self, **kwargs):
        super(UCB, self).initialize(**kwargs)
        print(self.model.gp)
        self._initial = True

    def acquisition(self, X):
        X = X.reshape(-1, self.domain.d)
        return -(self.model.ucb(X))

    def add_data(self, data):
        super().add_data(data)


    def acquisition_grad(self, x):
        mean, var = self.model.mean_var(x)
        std = np.sqrt(var)
        dmu_dX, dv_dX = self.model.mean_var_grad(x)
        dmu_dX = dmu_dX.reshape(dmu_dX.shape[0:2]) # flatten out inner dimension
        return -(mean + self.model.beta * std), -(dmu_dX + self.model.beta  * dv_dX) / (2 * std)


class SafeUCB(SafetyMixin, UCB):
    pass


# class SafeGPUCB(ModelAlgorithm):
#
#     def __init__(self, *args, **kwargs):
#         super(SafeGPUCB, self).__init__(*args, **kwargs)
#         self._additional_dtype_evaluation_fields = [('expanding', 'b'), ] # the numpy dtype '?' for bool does not seem to work with hpy5
#
#
#     def initialize(self, *args, **kwargs):
#         super(SafeGPUCB, self).initialize(*args, **kwargs)
#         self.current_safe_x = self.initial_evaluation.x
#
#     def get_next_evaluation_point(self):
#         expanding = np.random.binomial(1, p=self.config.SGPUCB_EXPAND_P)
#         if expanding:
#             self.logger.info("Expanding")
#             f = self.s_expand_grad if self.optimizer.requires_gradients else self.s_expand
#         else:
#             self.logger.info("GP_UCB")
#             f = self.sfun_grad if self.optimizer.requires_gradients else self.sfun
#
#
#         self.current_safe_x = self.optimizer.optimize(f)
#         return self.current_safe_x, {'expanding': expanding}
#
#     def sfun_grad(self, x):
#         x = x.reshape(1, -1)
#         mean, var = self.f.mean_var(x)
#         std = np.sqrt(var).squeeze()
#         dmu_dX, dv_dX = self.f.mean_var_grad(x)
#
#         # starting with standard acquisition function & gradient for ucb
#         y = -(mean + self.f.beta * std)
#         grad = -(dmu_dX.squeeze() + self.f.beta / std * dv_dX.squeeze())
#         safe = True
#
#         # add gradients of all safety gps which violate the safety constraint
#         for s in self.s:
#             s_mean, s_var = s.mean_var(x)
#             s_dmu_dX, s_dv_dX = s.mean_var_grad(x)
#             s_std = np.sqrt(s_var).squeeze()
#             s_ucb = s_mean.squeeze() + s.beta * s_std
#
#             # if safety constraint is violated
#             if s_ucb >= 0:
#                 grad += s_dmu_dX.squeeze() + s.beta / s_std * s_dv_dX.squeeze()
#                 y += s_ucb
#                 safe = False
#
#         return y, grad.squeeze(), safe
#
#     def s_expand_grad(self, x):
#         x = x.reshape(1, -1)
#         y = 0
#         grad = np.zeros(self.d)
#         safe = True
#
#         # add gradients of all safety gps which violate the safety constraint
#         for s in self.s:
#             s_mean, s_var = s.mean_var(x)
#             s_dmu_dX, s_dv_dX = s.mean_var_grad(x)
#             s_dmu_dX = s_dmu_dX.squeeze()
#             s_dv_dX = s_dv_dX.squeeze()
#             s_std = np.sqrt(s_var).squeeze()
#             s_ucb = s_mean.squeeze() + s.beta * s_std
#             s_ucb_grad = s_dmu_dX + s.beta / s_std * s_dv_dX
#
#             # if safety constraint is violated
#             if s_ucb >= 0:
#                 grad += s_ucb_grad
#                 y += s_ucb
#                 safe = False
#             s_var = s_var.squeeze()
#
#             bump_y, bump_grad = self.bump_grad(s_ucb)
#
#             y += -(s_var * bump_y)
#             grad += -(s_dv_dX * s_var + s_var * bump_grad * s_ucb_grad)
#
#         return y, grad.squeeze(), safe
#
#     def bump_grad(self, y):
#         A = 10
#         exp = np.exp(- A * np.square(y))
#         return exp, -2 * A * y * exp
#
#     def sfun(self, x):
#         y = - self.f.ucb(x.reshape(1, -1)).squeeze()
#         safe = True
#         for s in self.s:
#             s_ucb = s.ucb(x.reshape(1, -1)).squeeze()
#             # set the objective value to -1 on any unsafe action
#             if s_ucb > 0:
#                 safe = False
#                 y += s_ucb
#
#         return y, safe
#
#     def s_expand(self, x):
#         x = x.reshape(1, -1)
#         y = 0
#         safe = True
#
#
#         # add gradients of all safety gps which violate the safety constraint
#         for s in self.s:
#             s_mean, s_var = s.mean_var(x)
#             s_std = np.sqrt(s_var).squeeze()
#             s_ucb = s_mean.squeeze() + s.beta * s_std
#
#             # if safety constraint is violated
#             if s_ucb >= 0:
#                 y += s_ucb
#                 safe = False
#             s_var = s_var.squeeze()
#
#             bump_y, bump_grad = self.bump_grad(s_ucb)
#
#             y += -(s_var * bump_y)
#
#         return y, safe
#
#     # not used right now. possible barrier function which can be used with unconstraint optimization
#     def barrier(self, x):
#         return np.sum(- np.log(1 - x) - np.log(x))
#
#     def barrier_grad(self, x):
#         return 1 / (1 - x) + 1 / x
