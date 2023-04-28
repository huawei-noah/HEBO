import numpy as np
from febo.controller import ControllerConfig
from febo.environment import ContextMixin
from febo.utils import join_dtypes, get_logger, join_dtype_arrays
from febo.utils.config import ConfigField, assign_config
from .controller import Controller
from time import time
logger = get_logger("controller")

class SimpleControllerConfig(ControllerConfig):
    T = ConfigField(100, comment="Horizon")
    best_predicted_every = ConfigField(0, comment="Do .best_predict() on every n-th timestep, if set to 0, don't evaluate .best_predict()")

@assign_config(SimpleControllerConfig)
class SimpleController(Controller):
    # Controller to run one algorithm, on one environment, once

    def __init__(self, *args, **kwargs):
        super(SimpleController, self).__init__(*args, **kwargs)

        self.experiment = kwargs.get("experiment")
        self.algorithm = kwargs.get("algorithm")
        self.environment = kwargs.get("environment")
        self.dbase = kwargs.get("dbase", None)
        self.dset = None


        self.run_id = kwargs.get("run_id", None)
        self.group_id = kwargs.get("group_id", None)

        self.T = self.config.T
        self._exit = False
        self._data = [] # only used if self.dset is None


    def initialize(self, initialize_environment=True, algo_kwargs=None, run_id=None):
        self.run_id = run_id
        self._completed = False

        if initialize_environment:
            logger.info(f"Initializing environment: {self.environment.name}.")
            # initialize environment and get initial measurement
            env_info = self.environment.initialize()
            # merge env_info and algo_kwargs. give priority to algo_kwargs
            if algo_kwargs is None:
                algo_kwargs = {}
            for k, v in env_info.items():
                if not k in algo_kwargs:
                    algo_kwargs[k] = v

        # check if environment provides Tmax
        if not self.environment.Tmax is None:
            self.T = self.environment.Tmax
            logger.info(f"Setting T = Tmax = {self.T}.")
        # initialize algorithm
        logger.info(f"Initializing algorithm: {self.algorithm.name}.")
        self.algorithm.initialize(**algo_kwargs)

        best_predicted_fields = []
        if self.config.best_predicted_every > 0:
            # add all fields from environment.dtype except for 'x'
            best_predicted_fields += [(f'{field}_bp', d[0]) for field, d in self.environment.dtype.fields.items() if not field in ('y_max',)]
            if hasattr(self.algorithm, 'model'):
                best_predicted_fields += [('y_model_bp', 'f8'), ('y_std_model_bp', 'f8')]

        self._best_predicted_dtype = np.dtype(best_predicted_fields)
        self._time_dtype = np.dtype([('time_acq', 'f'), ('time_data', 'f')])
        self.evaluation_dtype = join_dtypes(self.algorithm.dtype, self.environment.dtype, self._best_predicted_dtype, self._time_dtype)

        self.t = 0

        # initialize dataset
        if not self.dbase is None:
            self.dset = self.dbase.get_dset(group=self.group_id, id=self.run_id, dtype=self.evaluation_dtype)
            self.run_id = self.dset.id

            group = self.dbase.get_group(self.group_id)
            group.attrs['environment'] =  self.environment.name
            group.attrs['algorithm'] = self.algorithm.name

            # load existing data
            if len(self.dset.data) == self.T:
                self.t = self.T
                self._completed = True

            elif len(self.dset.data) > 0:
                logger.info("Loading existing data into algorithm.")
                for evaluation in self.dset.data:
                    self.algorithm.add_data(evaluation)
                    self.t += 1

        self._data = []  # only used if self.dset is None
        self._exit = (self.t >= self.T)

        # set experiment_info data on algorithm
        self.algorithm.experiment_info["run_id"] = self.run_id
        self.algorithm.experiment_info["group_id"] = self.group_id


    def run(self):
        logger.info(f"Starting optimization: {self.algorithm.name}")
        # interaction loop
        while not self._exit:
            self._run_step()


    def finalize(self, finalize_environment=True):
        # finalize environment, but make sure we don't crash here, as we still need to close files in the controller

        if finalize_environment:
            try:
                logger.info("Finalizing environment...")
                self.environment.finalize()
            except Exception as e:
                logger.error("The following error occurred during finalizing the environment: %s" % str(e))

        logger.info(f"Finalizing: {self.algorithm.name}")

        if not self.dset is None:
            self.dset.adjust_size()

        return self.algorithm.finalize()

    def _handle_exception(self, e):
        raise e  # by default, just raise the exception

    def _run_step(self, stopping_Y=None):
        self.best_regret_reached = False
        logger.debug("Starting iteration %s" % self.t)
        try:
            evaluation = self._run_interaction(stopping_Y=stopping_Y)
            if self.config.best_predicted_every > 0:
                self._evaluate_best_predicted(evaluation)

            # record data
            if not self.dset is None:
                self.dset.add(evaluation)
            else:
                self._data.append(evaluation)  # if no dset is provided, manually record data

            self.t += 1
        except (Exception, KeyboardInterrupt) as e:
            self._handle_exception(e)

        if self.algorithm.exit or self.best_regret_reached:
            logger.info(f"Algorithm terminated.")

        self._exit = (self.t >= self.T) or self.algorithm.exit or self.best_regret_reached

    def _run_interaction(self, stopping_Y=None):
        if isinstance(self.environment, ContextMixin):
            # algorithm call with context
            context = self.environment.get_context()
            start = time()
            x, additional_data = self.algorithm.next(context)
            time_acq = time() - start
        else:
            # call algorithm without context
            start = time()
            x, additional_data = self.algorithm.next()
            time_acq = time() - start
        # logger.info(f"target from algorithm: {evaluation.x_target}")  # todo denormalize

        env_evaluation = self.environment.evaluate(x)

        evaluation = join_dtype_arrays(env_evaluation, additional_data, self.evaluation_dtype).view(np.recarray)

        start = time()
        self.algorithm.add_data(evaluation)
        time_data = time() - start

        evaluation['time_acq'] = time_acq
        evaluation['time_data'] = time_data

        # if self.t > 5:
        #     self.algorithm.optimize_model()

        logger.debug(f"Objective value {evaluation.y}.")
        # logger.info(f"safety constraints {evaluation.s}")
        logger.debug(f"Completed step {self.t}.")

        if type(stopping_Y)!=type(None):
            if stopping_Y == evaluation.y:
                self.best_regret_reached = True

        return evaluation

    def _evaluate_best_predicted(self, evaluation, **add_env_kwargs):
        if self.t % self.config.best_predicted_every == 0:
            best_predicted_x = self.algorithm.best_predicted()
            logger.debug(f"Calculating the best predicted point at: {best_predicted_x}")
            self._best_predicted_eval = self._evaluate_best_predicted_environment(best_predicted_x, **add_env_kwargs)

        # copy values from environment evaluation at x_best_predicted to evaluation
        for field in self._best_predicted_eval.dtype.fields:
            bp_field_name = f'{field}_bp'
            if bp_field_name in self.evaluation_dtype.fields:
                    evaluation[bp_field_name] = self._best_predicted_eval[field]

        # if algorithm uses a model, also evalute the model at best_predicted_x
        if hasattr(self.algorithm, 'model'):
            m,v = self.algorithm.model.mean_var(evaluation['x_bp'])
            evaluation['y_model_bp'] = m
            evaluation['y_std_model_bp'] = np.sqrt(v)

    def _evaluate_best_predicted_environment(self, x, **kwargs):
        return self.environment.evaluate(x,  **kwargs)