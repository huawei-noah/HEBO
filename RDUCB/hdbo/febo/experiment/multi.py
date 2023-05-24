from febo.utils import locate
from febo.experiment.simple import SimpleExperimentConfig, SimpleExperiment
from febo.utils.config import ConfigField, assign_config, config_manager, ClassConfigField
import os
from ruamel.yaml import YAML
import yaml

class SubconfigField(ConfigField):

    def _decode(self, value):
        if isinstance(value, str):
            value = locate(value)()

        decoded_list = []
        for benchmark in value:
            benchmark_settings = {}
            for section, settings in benchmark.items():

                # support for flattend syntax '{section:setting : value }'
                if ':' in section:
                    section, setting = section.rsplit(':')
                    settings = { setting : settings}

                for setting, value in settings.items():
                    if not section in benchmark_settings:
                        benchmark_settings[section] = {}

                    if section in config_manager._fields and setting in config_manager._fields[section]:
                        field = config_manager._fields[section][setting]
                        benchmark_settings[section][setting] = field._encode(value)

            decoded_list.append(benchmark_settings)
        return decoded_list

    # def _encode(self, value):
    #     encoded_list = []
    #     for batch in value:
    #         batch_dict = {}
    #         for section, settings in batch.items():
    #             if not section in batch_dict:
    #                 batch_dict[section] = {}
    #
    #             for setting, value in settings.items():
    #                 batch_dict[section][setting] = self._config_manager._fields[section][setting]._encode(value)
    #
    #         encoded_list.append(batch_dict)
    #
    #     return  encoded_list

def label_id(id, config):
    return id

class MultiExperimentConfig(SimpleExperimentConfig):
    fixed_environment = ConfigField(False, comment='If true, only one environment for the whole batch will be created. Use this, if you randomly genrate your environment, but the whole batch should use the same random instance of the environment.')
    iterator = SubconfigField({})
    multi_controller = ClassConfigField('febo.controller.multi.RepetitionController')
    label = ClassConfigField(label_id)
    _section = 'experiment.multi'


class ExperimentPart:
    def __init__(self, base_dir, id, label_fun):
        self._config = None
        self._id = id
        self._path = os.path.join(base_dir, str(id))
        self._label_fun = label_fun
        self._label = None
        self._controller = None

    @property
    def id(self):
        return self._id

    @property
    def config(self):
        return self._config

    @property
    def label(self):
        return self._label

    @property
    def controller(self):
        return self._controller

    @property
    def path(self):
        return self._path

    def set_config(self, config):
        self._config = config
        self._label = self._label_fun(self.id, self.config)

    def set_controller(self, controller):
        self._controller= controller

    def apply_config(self, update_configs=[]):
        config_manager.load_data(self._config)
        for config in update_configs:
            config_manager.update_config(config)

    def save(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        with open(os.path.join(self.path, 'benchmark.yaml'), 'w') as f:
            yaml = YAML()
            yaml.dump(self._config, f)

    def load(self):
        with open(os.path.join(self.path, 'benchmark.yaml'), 'r') as f:
            self._config = yaml.load(f)
        self._label = self._label_fun(self.id, self.config)


@assign_config(MultiExperimentConfig)
class MultiExperiment(SimpleExperiment):
    def __init__(self, experiment_dir):
        super().__init__(experiment_dir)
        self._parts = []
        self.environment = None

    @property
    def parts(self):
        return self._parts

    def create(self, name=None):
        # Do not run SimpleExperiments create here
        super(SimpleExperiment, self).create(name)
        batch_dir = os.path.join(self._directory, "parts")
        os.mkdir(batch_dir)

        self._parts = []

        if self.config.fixed_environment:
            self.environment = self.config.environment(path=self.directory)

        for batch_id, config in enumerate(self.config.iterator):
            part = ExperimentPart(batch_dir, batch_id, self.config.label)
            part.set_config(config)
            self._parts.append(part)
            part.save()
            part.apply_config(update_configs=[self.config])
            if not self.config.fixed_environment:
                self.config.environment(path=part.path)

    def load(self, name):
        # Do not run SimpleExperiment's load
        super(SimpleExperiment, self).load(name)
        batch_dir = os.path.join(self._directory, "parts")
        self._parts = []

        # read all subdirectories in the 'parts' folder
        for folder in sorted(os.listdir(batch_dir), key=lambda e : int(e)):
            if os.path.isdir(os.path.join(batch_dir, folder)):
                batch_id = int(folder)
                part = ExperimentPart(batch_dir, batch_id, self.config.label)
                part.load()
                self._parts.append(part)

        # create environment already here, if it is fixed for the experiment
        if self.config.fixed_environment:
            self.environment = self.config.environment(path=self.directory)


    def get_controller(self, remote=False):
        """
        returns a controller instance
        :param kwargs: arguments passed to the controller
        :return: controller instance
        """
        if self.config.fixed_environment:
            environment = self.environment

        # create a sub-controller for each part
        for part in self._parts:
            part.apply_config(update_configs=[self.config])

            if not self.config.fixed_environment:
                environment = self.config.environment(path=part.path)

            algorithm = self.config.algorithm(experiment_dir=self.directory)
            controller = super().get_controller(remote=remote,
                                                environment=environment,
                                                algorithm=algorithm,
                                                group_id=part.id)

            part.set_controller(controller)

        return self.config.multi_controller(parts=self.parts, fixed_environment=self.environment)
