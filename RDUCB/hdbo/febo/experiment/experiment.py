import os
from febo import utils
from febo.experiment.data import DataBase
from febo.utils import get_timestamp, get_logger
from febo.utils.config import Config, ConfigField, config_manager, assign_config, Configurable
try:
    import ceramo
except ImportError:
    pass

logger = get_logger("experiment")

class ExperimentConfig(Config):
    _section = 'experiment'


@assign_config(ExperimentConfig)
class Experiment(Configurable):
    def __init__(self, experiment_dir):
        self._directory = None
        self._timestamp = None
        self._dbase = None
        self._experiment_dir = experiment_dir

    @property
    def directory(self):
        return self._directory

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def name(self):
        return self._experiment_name

    @property
    def hdf5(self):
        """ direct access to hdf5 data """
        return self._dbase.hdf5

    def path(self, name):
        return os.path.join(self._experiment_dir, name)

    def exists(self, name):
        return os.path.exists(self.path(name))

    def create(self, name):
        """
            Creates a new experiment with name `experiment_name`.
            Info `experiment_name == "` no directory inside `experiment_dir` (as passed to the constructor) is created.
        """
        self._experiment_name = name
        if not os.path.exists(self._experiment_dir):
            raise Exception(f"Experiment directory '{self._experiment_dir}' does not exist.")

        self._directory = os.path.join(self._experiment_dir, name)

        # accept empty experiment name, in this case, directly write to experiment_dir
        if not name == '':
            if os.path.exists(self._directory):
                raise Exception(f"Could not create experiment, directory {self._directory} already exists.")

            os.mkdir(self._directory)

        get_logger.set_path(self._directory)
        self._save_config_file()
        logger.info(f'Created a new experiment in {self._directory}.')

        self._dbase = DataBase(self._directory)
        self._timestamp = get_timestamp()
        self._dbase.create()
        self._dbase.hdf5.attrs['timestamp'] = self._timestamp
        self._dbase.close()

    def load(self, name):
        """
            Load existing experiment
            path: Path where experiment is stored
        """
        self._experiment_name = name
        full_path = os.path.join(self._experiment_dir, name)
        if not os.path.exists(full_path):
            raise Exception("Could not find experiment directory (%s)." % full_path)

        self._directory = full_path

        # set path for log files
        get_logger.set_path(self._directory)

        # load config and update own config
        config_manager.load_yaml(os.path.join(self._directory, 'experiment.yaml'))
        config_manager.update_config(self.config)

        self._dbase = DataBase(self._directory)
        self._dbase.open()
        if not 'timestamp' in self._dbase.hdf5.attrs:
            logger.warning("timestamp not set, adding now.")
            self._dbase.hdf5.attrs['timestamp'] = get_timestamp()
        self._timestamp = self._dbase.hdf5.attrs['timestamp']
        logger.info(f'Loaded experiment from {self._directory}.')

    def get_controller(self, remote=False, **kwargs):
        """
        returns a controller instance
        :param kwargs: arguments passed to the controller
        :return: controller instance
        """

        raise NotImplementedError

    def start(self, remote=False):
        controller = self.get_controller(remote=remote)
        try:
            controller.initialize()
            controller.run()
        finally:
            controller.finalize() # important for file closing

    def close(self):
        if not self._dbase is None:
            self._dbase.close()

    def _save_config_file(self, path=None):
        if path is None:
            path = self._directory
        config_path = utils.join_path_if_not_exists(path, 'experiment.yaml')

        config_manager.write_yaml(config_path)


    def sync(self, sync_dir):
        """

        Returns:

        """
        with ceramo.results.Backend() as backend:
            results = backend.query({'task': 'febo.controller.remote.run_simple_controller', 'status': 'SUCCESS',
                                     'experiment_name' : self._experiment_name,
                                     'experiment_timestamp' : self._timestamp})

        return ceramo.results.load_remote_data(results, sync_dir, verbose=False)




