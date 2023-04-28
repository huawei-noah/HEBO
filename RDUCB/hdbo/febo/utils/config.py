import inspect
from enum import Enum

from io import StringIO
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from febo.utils import locate, get_logger

logger = get_logger('config')

class ConfigManager:
    """ class to track all instances of Config classes, and load and save configs to yaml. """
    def __init__(self):
        self._configs = []  # list of all configs
        self._config_classes = []  # list of all config classes
        self._configurable_classes = []  # list of all config classes
        self._fields = {}  # dictionary of all fields
        self._fields_origin = {}  # dictionary to keep track what class fields come from
        self._data = {}   # dict for current data
        self._locked = False
        self._autoregister = False

    def reset(self):
        for key in self._data.keys():
            self._data[key] = {}

    def set_autoregister(self, autoregister):
        self._autoregister = autoregister

    @property
    def autoregister(self):
        return self._autoregister

    def register(self, *args):
        """
        Register a config class to the config manager

        Args:
            *args: config_cls1, config_cls2, ...

        """
        for config_cls in args:
            # mixin configs don't inherit from Config, hence we also test for _section attribute
            if issubclass(config_cls, Config) or hasattr(config_cls, '_section'):
                self._register_config(config_cls)
            elif issubclass(config_cls, Configurable):
                self._register_configurable(config_cls)

    def _register_configurable(self, configurable_cls):

        if hasattr(configurable_cls, '_config_class'):
            self._register_config(getattr(configurable_cls, '_config_class'))

        if hasattr(configurable_cls, '_config_mixin'):
            self._register_config(getattr(configurable_cls, '_config_mixin'))

        self._configurable_classes.append(configurable_cls)

        # recurse on base classes
        for base_cls in configurable_cls.__bases__:
            if base_cls not in self._configurable_classes and base_cls not in (Configurable, object):
                self._register_configurable(base_cls)


    def _register_config(self, config_cls):
        if config_cls in self._config_classes:
            return

        if not hasattr(config_cls, '_section'):
            raise Exception("Every config class has to define a '_section' attribute.")

        self._config_classes.append(config_cls)

        section = config_cls._section
        for name, field in vars(config_cls).items():
            if not isinstance(field, ConfigField):
                continue  # skip attributes which are not a ConfigField

            # add section to self._fields and self._data
            if not section in self._fields:
                self._fields[section] = {}
                self._fields_origin[section] = {}
                self._data[section] = {}

            # enforce unique names among sections
            if name in self._fields[section]:
                if f"{config_cls.__module__}.{config_cls.__name__}" != self._fields_origin[section][name]:
                    raise Exception(f"A field with name '{name}' was already registered in section '{section}' by a different config class.")

            self._fields[section][name] = field
            self._fields_origin[section][name] = f"{config_cls.__module__}.{config_cls.__name__}"

    def register_instance(self, config):
        """
        Register a new config instance.

        Args:
            config: the config instance

        """
        self._configs.append(config)

    def set_setting(self, section, setting, value):
        if not section in self._fields:
            raise Exception(f"Unknown section '{section}'.")

        if not setting in self._fields[section]:
            raise Exception(f"Unknown setting '{setting}' in section {section}.")

        self._data[section][setting] = self._fields[section][setting]._decode(value)

    def get_setting(self, section, setting):
        if not section in self._fields:
            raise Exception(f"Unknown section '{section}'.")

        if not setting in self._fields[section]:
            raise Exception(f"Unknown setting '{setting}' in section {section}.")

        field = self._fields[section][setting]
        return field._encode(self._data[section].get(setting, field.default))

    def load_data(self, data, update_existing_instances=False):
        """
        Load data into the ConfigManager. This data is then used for new Config instances.

        Args:
            data (dict): data dictionary
            update_existing_instances: if True, all registered instances will be updated with the new data

        """
        # copy new data into self._data
        for section, section_data in data.items():
            if not section in self._fields:
                logger.warning(f"Recieved setting data for unregistered section {section}.")
                self._data[section] = {}


            for setting, value in section_data.items():
                if section in self._fields and not setting in self._fields[section]:
                    logger.warning(f"Recieved setting data for unregistered setting {setting}.")

                self._data[section][setting] = value

        if update_existing_instances:
            # go through registered sections and associated configs

            for config in self._configs:
                for section in data:
                    for setting, value in data[section].items():
                        if section in config._sections and hasattr(config, setting):
                            setattr(config, setting, value)


    def update_config(self, config):
        for name, field in config._fields.items():
            section = field._section
            if section in self._data and name in self._data[section]:
                setattr(config, name, self._data[section][name])

    def load_yaml(self, file, update_existing_instances=True, section=None):
        """
        Loads the configuration from file and updates all instances.

        Args:
            file:
            section: if not None, load only this section

        Returns:

        """
        yaml = YAML()
        with open(file, 'r') as stream:
            data = yaml.load(stream)

        # if section is given, filter data to this section
        if section is not None and section in data:
            data = {section : data[section]}

        self.load_data(data, update_existing_instances=update_existing_instances)
        return dict(data)

    def write_yaml(self, file, include_default=False):
        """
        Dumps the current config to the yaml file.

        Args:
            file:

        Returns:

        """
        yaml = YAML()
        yaml.default_flow_style = False
        with open(file, 'w') as stream:
            yaml.dump(self._get_data_encoded(include_default=include_default), stream)



    def get_yaml(self, include_default=False):
            """
            Returns (str): The current configuration as yaml.

            """
            yaml = YAML()
            yaml.default_flow_style = False
            ss = StringIO()


            yaml.dump(self._get_data_encoded(include_default=include_default), ss)
            s = ss.getvalue()
            ss.close()
            return s

    def _get_data_encoded(self, include_default=False):
        data_encoded = {}
        for section in self._fields:

            if not include_default and (not section in self._data or len(self._data[section]) == 0):
                continue

            data_encoded[section] = CommentedMap()

            for setting, field in self._fields[section].items():
                if setting in self._data[section] or include_default:
                    data_encoded[section].insert(0, setting, field._encode(self._data[section].get(setting, field.default)), field.comment)

        data_encoded = CommentedMap(sorted(data_encoded.items(), key=lambda t: t[0]))

        return data_encoded

    def get_config_as_dict(self, include_default=False):
        config_dict = {}
        for section in self._fields:

            if not include_default and (not section in self._data or len(self._data[section]) == 0):
                continue

            config_dict[section] = {}

            for setting, field in self._fields[section].items():
                if setting in self._data[section] or include_default:
                    config_dict[section][setting] = field._encode(self._data[section].get(setting, field.default))

        return config_dict

def all_subconfig(config_cls):
    """
    Creates a list of all config classes in the hereditary tree.

    Args:
        config_cls (type): the root config class

    Returns (list): list of config classes

    """
    # stop on base Config class or object
    if config_cls is object or config_cls is Config:
        return []

    configs = [config_cls]
    for subcls in config_cls.__bases__:
        configs += all_subconfig(subcls)  # recurse

    return configs



class Config:
    """
    Config base class.
    """

    def __new__(cls, *args, **kwargs):

        # quick check if the _section attribute is defined
        instance = super().__new__(cls)
        instance._fields = {}
        instance._field_values = {}
        instance._sections = []

        for subclass in all_subconfig(cls):
            if not hasattr(subclass, "_section"):
                raise Exception(f'The config class "{subclass}" does not define a "_section" class attribute.')

            section = subclass._section
            for name, field in vars(subclass).items():
                if isinstance(field, ConfigField):
                    if not name in instance._fields: # child class might overwrite settings
                        field._section = section
                        field._config_cls = subclass
                        instance._fields[name] = field
                        instance._field_values[name] = field.default

            if not section in instance._sections:
                instance._sections.append(section)


        return instance

    def __getattribute__(self, key):
        """ redirect all non-private attributes which are in self._attr_names to self._fields """
        if not key.startswith('_') and key in self._field_values:
            return self._field_values[key]
        else:
            return super(Config, self).__getattribute__(key)

    def __setattr__(self, key, value):
        """ update all non-private attributes on fields """
        if not key.startswith('_'):
            if key in self._field_values:
                self._field_values[key] = self._fields[key]._decode(value)
            else:
                raise AttributeError(f'Field "{key}" not found in config "{self.__class__}".')
        else:
            super(Config, self).__setattr__(key, value)

    def __dir__(self):
        return super().__dir__() + self._fields.keys()

    def _update_data(self, data):
        for k, v in data.items():
            setattr(self, k, v)

def identitiy(o):
    return o

class ConfigField:
    """
    Field for config.
    """
    def __init__(self, default, field_type=identitiy, comment=None, allow_none=False):
        """
        Args:
            field_type:
            default:
            comment:
        """
        self._allow_none = allow_none
        if field_type is not None:
            self._field_type = field_type
            if allow_none and default is None:
                self._default = None
            else:
                self._default = field_type(default)

        else:
            self._default = default
            self._field_type = type(default)

        self._comment = comment if comment else None # set empty string as None (to avoid ruaml CommentedMap to fail)
        self._section = None  # set later by Config.__new__
        self._config_cls = None  # set later by Config.__new__

    def _encode(self, value):
        return value

    def _decode(self, value):
        if self._allow_none and value is None:
            return None

        return self._field_type(value)

    @property
    def default(self):
        return self._decode(self._default)

    @property
    def comment(self):
        return self._comment

    @property
    def section(self):
        return self._section

class ClassConfigField(ConfigField):
    def _encode(self, value):
        if self._allow_none and value is None:
            return None

        if isinstance(value, str):
            return value

        return f"{value.__module__}.{value.__name__}"

    def _decode(self, value):
        if self._allow_none and value is None:
            return None
        if isinstance(value, str):
            return locate(value)

        return value

class ClassListConfigField(ConfigField):
    def _encode(self, value):

        if self._allow_none and value is None:
            return None


        list_encoded = []
        for item in value:
            if isinstance(item, str):
                list_encoded.append(item)
            else:
                list_encoded.append(f"{item.__module__}.{item.__name__}")

        return list_encoded

    def _decode(self, value):
        if self._allow_none and value is None:
            return None

        list_decoded = []
        for item in value:
            if isinstance(item, str):
                list_decoded.append(locate(item))
            else:
                list_decoded.append(item)

        return list_decoded

class EnumConfigField(ConfigField):

    def __init__(self, default, enum_cls, field_type=identitiy, comment=None, allow_none=False):
        self._enum_cls = enum_cls
        super().__init__(default, field_type, comment, allow_none)

    def _decode(self, value):
        if isinstance(value, self._enum_cls):
            return value
        return self._enum_cls[value]

    def _encode(self, value):
        if isinstance(value, Enum):
            return str(value).split('.')[1]

        return value


config_manager = ConfigManager()  # config manager

def assign_config(config_cls, config_manager=config_manager):
    """
    Decorator to assign configuration classes.

    Args (Config):
        config_cls: Configuration class which is assigned.
    """

    # we use a call to assign_config to register a default config

    # decorator
    def decorator(cls):
        if issubclass(config_cls, Config):
            cls._config_class = config_cls
        else:
            cls._config_mixin = config_cls

        cls._config_manager = config_manager

        if config_manager.autoregister:
            print(f"auto register {cls}")
            config_manager.register(cls)
        return cls
    return decorator


def _configure_mixin_configs(cls):
    """
    Recursively goes through all bases of cls, and creates the same hierachy structure for the configs,
    for the case where there are mixin classes.

    Args:
        cls (type):


    """
    if cls is Configurable or cls is object:
        return Config

    # last class is proper base class
    super_cls = cls.__bases__[-1]

    # recurse on proper base class (not on mixins)
    super_cls_config = _configure_mixin_configs(super_cls)

    config_mixins = []
    # go through mixin classes and check if mixin has a _config_class defined
    # if yes, added this mixin_config as mixin to config_cls
    for mixin in cls.__bases__[:-1]:
        if hasattr(mixin, "_config_mixin"):
            config_mixins.append(mixin._config_mixin)

    # only if there where config_mixins found, add them to main config class as mixins
    if len(config_mixins):
        # get _config_class attribute or create an EmptyConfig as fallback,
        config_cls = getattr(cls, "_config_class", type(f"__{cls.__name__}Config", (super_cls_config,), {}))
        config_cls.__bases__ = tuple([m for m in config_mixins if not m in config_cls.__bases__]) \
                               + config_cls.__bases__

        # reassign config_class
        cls._config_class = config_cls
    if hasattr(cls, "_config_class"):
        return cls._config_class

    return super_cls_config


class Configurable:
    """ Base class for configurable classes """

    def __new__(cls, *args, **kwargs):
        """
        Configure the config class (account for mixins) and add a "config" instance variable.
        """
        _configure_mixin_configs(cls)
        instance = super().__new__(cls)
        if hasattr(instance, '_config_class'):
            instance.config = cls._config_class()
            config_manager.register_instance(instance.config)
            config_manager.update_config(instance.config)

        return instance



