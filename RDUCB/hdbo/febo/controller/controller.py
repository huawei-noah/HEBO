from febo.utils import get_logger
from febo.utils.config import ConfigField, Config, Configurable, assign_config


class ControllerConfig(Config):
    _section = 'controller'

logger = get_logger("controller")

@assign_config(ControllerConfig)
class Controller(Configurable):
    def __init__(self, *args, **kwargs):
        self._completed = False

    def initialize(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def finalize(self):
        pass

    @property
    def completed(self):
        return self._completed