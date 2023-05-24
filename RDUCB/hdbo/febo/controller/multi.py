from febo.controller import ControllerConfig
from febo.utils import get_logger
from febo.utils.config import config_manager, assign_config, ConfigField
from .controller import Controller

logger = get_logger("controller")


class MultiController(Controller):
    # base class to run multiple algorithms, instantiated by MultiExperiment.

    def __init__(self, parts, fixed_environment=None):
        super(MultiController, self).__init__()
        self.parts = parts
        self._fixed_environment = fixed_environment
        self._algo_kwargs = None

    def initialize(self):
        # initialize environment if fixed_environment is provided
        if not self._fixed_environment is None:
            self._algo_kwargs = self._fixed_environment.initialize()

    def run(self):
            for part in self.parts:
                self._run_part(part)
                if hasattr(part.controller, '_global_exit'):
                    if part.controller._global_exit:
                        break


    def _run_part(self, part):
        """
        Stub to be overwritten by subclasses. Should run the subexperiment given by part (initialize,run,finalize).
        Args:
            part:

        Returns:

        """
        raise NotImplementedError

    def finalize(self):
        # initialize environment if fixed_environment is provided
        if not self._fixed_environment is None:
            self._algo_kwargs = self._fixed_environment.finalize()

class RepetitionControllerConfig(ControllerConfig):
    repetitions = ConfigField(5, comment='Number of repetitions each experiment is run.')

@assign_config(RepetitionControllerConfig)
class RepetitionController(MultiController):
    """
    Controller that runs multiple subcontrollers `config.repetition` times.
    """

    def _run_part(self, part):
        part.apply_config()
        # run all repetitions of using same controller
        for rep in range(self.config.repetitions):
            part.controller.initialize(initialize_environment=(self._fixed_environment is None),
                                       algo_kwargs=self._algo_kwargs,
                                       run_id=rep)  # explicitly set run_id

            if part.controller.completed:
                logger.info("Run complete, skipping.")
            else:
                part.controller.run()

            part.controller.finalize(finalize_environment=(self._fixed_environment is None))


class SequentialController(MultiController):
    """
    Controller to run multiple algorithms in sequence.
    Each algorithm is initialized with the dict return by the previous algorithms finalize().
    """

    def __init__(self, parts, fixed_environment=None):
        if fixed_environment is None:
            raise Exception("Can run sequential controller only with a fixed environment for now.")
        super().__init__(parts=parts, fixed_environment=fixed_environment)
        self._run_id = None

    def _run_part(self, part):
        part.apply_config()

        # if we already have a run_id, set it on the controller to make sure, all parts get the same id
        part.controller.initialize(algo_kwargs=self._algo_kwargs, initialize_environment=False, run_id=self._run_id)

        # if we do not have a run_id yet, take it from the controller
        if self._run_id is None:
            self._run_id = part.controller.run_id

        part.controller.run()
        finalize_result = part.controller.finalize(finalize_environment=False)

        # update algorithm kwargs
        if not finalize_result is None:
            for key,item in finalize_result.items():
                self._algo_kwargs[key] = item

        return finalize_result
