import numpy as np
from febo.utils import get_logger, locate
from febo.solvers import ScipySolver
logger = get_logger('algorithm')

class SafetyMixin:


    def initialize(self, **kwargs):
        super(SafetyMixin, self).initialize(**kwargs)

        self.expander_solver = self._get_solver(self.domain)


        # set safe_mode on optimizer
        self.expander_solver.safe_mode = True
        self.solver.safe_mode = True

        self.expander_solver.safety_wrapper = self.safety_wrapper
        self.solver.safety_wrapper = self.safety_wrapper

        # disable exception when no feasible point was found
        self.expander_solver.infeasible_exception = False
        self.solver.infeasible_exception = False


    def safety_wrapper(self, x, y):
        safe = [True] * len(x)
        # add gradients of all safety gps which violate the safety constraint
        for s in self.s:
            s_mean, s_var = s.mean_var(x)
            s_std = np.sqrt(s_var)
            s_ucb = s_mean + s_std

            # add ucb of any violated safety constraint
            for i, ucb_i in enumerate(s_ucb):
                if ucb_i > 0:
                    safe[i] = False
                    y[i] += ucb_i

            bump_y = self.bump(s_ucb)
            y += -(s_std/s.scale * bump_y)

        return y, safe


    def next(self):
        x, info = super(SafetyMixin, self).next()
        info['point_type'] = self._point_type
        return x, info

    def _next(self):
        self._point_type = 'initial'
        if self.t == 0:
            return self.x0

        self.acquisition_init()
        x_acq,_ = self.solver.minimize(self.acquisition)
        x_exp, _ = self.expander_solver.minimize(self.expander_acquisiton)

        if x_acq is None and x_exp is None:
            logger.warning('Failed to find feasible point. Choosing initial parameter.')
            return self.x0


        if x_acq is None:
            logger.warning('Failed to find feasible acquisition point. Choosing expander')
            return x_exp

        if x_exp is None:
            logger.warning('Failed to find feasible expander. Choosing acquisition point.')
            return x_acq

        var_x_acq = self._get_uncertainty(x_acq)
        var_x_exp = self._get_uncertainty(x_exp)

        if var_x_acq >= var_x_exp:
            logger.info('Choosing acquisition point')
            self._point_type = 'acquisition'
            return x_acq
        else:
            logger.info('Choosing expander')
            self._is_expander = True
            self._point_type = 'expander'
            return x_exp


    def _get_uncertainty(self, x):
        var = self.model.var(x)/self.model.gp.kern.variance

        for m in self.s:
            var = max(m.var(x)/m.gp.kern.variance, var)
        return var

    def expander_acquisiton(self, x):
        x = np.atleast_2d(x)
        y = np.zeros(shape=(x.shape[0],1))


        for s in self.s:
            s_mean, s_var = s.mean_var(x)
            s_std = np.sqrt(s_var)
            s_ucb = s_mean + s_std

            bump_y = self.bump(s_ucb)
            y += -(s_var/s.scale * bump_y)

        return y

    def bump(self, y):
        A = 5
        return A * np.exp(- np.square(y))

    def _is_safe(self, x):
        """
        helper function for plotting
        """
        safe = [True] * len(x)
        for s in self.s:
            s_mean, s_var = s.mean_var(x)
            s_std = np.sqrt(s_var)
            s_ucb = s_mean + s_std

            # add ucb of any violated safety constraint
            for i, ucb_i in enumerate(s_ucb):
                if ucb_i > 0:
                    safe[i] = False

        return np.array(safe, dtype=float)


    @property
    def requires_x0(self):
        return True

    def _get_dtype_fields(self):
        """
        Fields used to define ``self.dtype``.

        Returns:

        """
        fields = super(SafetyMixin, self)._get_dtype_fields()
        return fields + [('point_type', 'S25')]