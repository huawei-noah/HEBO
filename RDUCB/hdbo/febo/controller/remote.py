import numpy as np
from febo.controller import SimpleController

from febo.experiment import SimpleExperiment
from febo.utils import get_logger
from febo.utils.config import config_manager
from febo.main import celery_app



logger = get_logger("controller")

class RemoteController(SimpleController):

    def __init__(self, *args, **kwargs):
        super(RemoteController, self).__init__(*args, **kwargs)


    def initialize(self, **kwargs):
        # first check if data table for given run_id exists already
        # we don't allow this for now
        run_id = kwargs.get('run_id', None)
        self._dset_exists = not run_id is None and self.dbase.dset_exists(group=self.group_id, id=run_id)

        super().initialize(**kwargs)

        self._initialize_kwargs = kwargs
        self._config = config_manager.get_config_as_dict()

        # check if we have a dbase object
        if self.dbase is None:
            raise Exception("Need database for remote controller.")

        self.dset.adjust_size()
        self.run_id = self.dset.id

    def run(self):
        if self._dset_exists:
            logger.warning("dset already exists, not submitting.")
        else:
            logger.info("Starting a remote controller using ceramo.")
            seed = int(np.random.randint(2**32-1, dtype=np.uint32))
            run_simple_controller.delay(self._config, self.experiment.name, self.experiment.timestamp, self.run_id, self.group_id, self._initialize_kwargs, seed)

    def finalize(self, finalize_environment=True):
        super().finalize(finalize_environment=True)


try:
    import ceramo

    @ceramo.task(celery_app, checkpoint=True)
    def run_simple_controller(config, experiment_name, experiment_timestamp, run_id, group_id, initialize_kwargs, seed, checkpoint=None):
        """
        run a simple controller

        Args:
            config:
            experiment_name:
            run_id:
            group_id:
            initialize_kwargs:
            checkpoint:

        Returns:

        """
        np.random.seed(seed)

        # make sure experiment.simple:controller is set to SimpleController
        if not 'experiment.simple' in config:
            config['experiment.simple'] = {}
        config['experiment.simple']['controller'] = 'febo.controller.SimpleController'
        config_manager.reset()
        config_manager.load_data(config)

        # create a simple experiment
        experiment = SimpleExperiment(checkpoint.directory)
        experiment.create(name='') # empty experiment_name, ie write directly to experiment_dir
        experiment.load(name='')
        checkpoint.store(experiment_name=experiment_name)
        checkpoint.store(experiment_timestamp=experiment_timestamp)


        # manually run controller to account for additional kwargs which are not set by SimpleExperiment
        controller = experiment.get_controller(run_id=run_id, group_id=group_id)
        controller.initialize(**initialize_kwargs)
        controller.run()
        controller.finalize()
        experiment.close()
except ImportError:
    pass
