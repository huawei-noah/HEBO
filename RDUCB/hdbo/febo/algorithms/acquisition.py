from febo.algorithms import AlgorithmConfig, Algorithm
from febo.environment import DiscreteDomain
from febo.utils import get_logger
from febo.utils.config import ClassConfigField, assign_config, ConfigField
from febo import solvers

logger = get_logger('algorithm')

class AcquisitionAlgorithmConfig(AlgorithmConfig):
    solver = ClassConfigField(None, field_type=str, allow_none=True)
    evaluate_x0 = ConfigField(True)
    _section = 'algorithm.acquisition'


@assign_config(AcquisitionAlgorithmConfig)
class AcquisitionAlgorithm(Algorithm):
    """
    Algorithm which is defined through an acquisition function.
    """

    def initialize(self, **kwargs):
        super(AcquisitionAlgorithm, self).initialize(**kwargs)
        self._evaluate_x0 = self.config.evaluate_x0

        self.solver = self._get_solver(domain=self.domain)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['solver']
        return self_dict

    def acquisition(self, x):
        raise NotImplementedError

    def acquisition_init(self):
        pass

    def acquisition_grad(self, x):
        raise NotImplementedError

    def _next(self, context=None):
        if self._evaluate_x0:
            self._evaluate_x0 = False
            if self.x0 is None:
                logger.error("Cannot evaluate x0, no initial point given")
            else:
                logger.info(f"{self.name}: Choosing initial point.")
                return self.x0

        # for contextual bandits, if domain changes, adjust solver (for now, just a new instance)
        if not context is None and 'domain' in context:
            self.solver = self._get_solver(context['domain'])

        self.acquisition_init()

        if self.solver.requires_gradients:
            acq = self.acquisition_grad
        else:
            acq = self.acquisition
        x, _ = self.solver.minimize(acq)
        return x

    def _get_solver(self, domain):
        if not self.config.solver is None:
            solver = self.config.solver(domain=domain, initial_x=self.x0)
        else:
            # if solver is not provided, use default choices
            if isinstance(domain, DiscreteDomain):
                solver = solvers.FiniteDomainSolver(domain=domain)
            else:
                solver = solvers.ScipySolver(domain= domain, initial_x=self.x0)
        return solver
