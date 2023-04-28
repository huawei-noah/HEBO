import sys
import os
import shutil

from febo.utils import locate, get_logger, get_timestamp
import argparse
from febo import utils
from febo.utils.config import Config, ConfigField, config_manager, ClassConfigField
from celery.signals import celeryd_after_setup
from celery import Celery

try:
    import ceramo

    celery_app = Celery('tasks')
    celery_app.config_from_object(ceramo.celeryconfig)
except ImportError:
    pass


# if subdirectory is in path, remove it there, because this causes trouble with relative imports
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir in sys.path:
    sys.path.pop(sys.path.index(current_dir))
# also remove relatives path which are added by the profiler
if 'febo' in sys.path:
    sys.path.pop(sys.path.index('febo'))

logger = get_logger('main')

class MainConfig(Config):
    experiment = ClassConfigField('febo.experiment.SimpleExperiment', comment="Experiment")
    modules = ConfigField([])
    log_level_console = ConfigField('INFO')
    log_level_file = ConfigField('INFO')
    experiment_dir = ConfigField('runs/')
    sync_dir = ConfigField('remote/')
    plotting_backend = ConfigField(None, allow_none=True, comment='Set to "agg" on machines where default matplotlib qt backend is not available.')
    _section = 'main'


main_config = MainConfig()


# config_manager.register_instance(config)


def main():
    """
    Main program entry point.
    """
    config_manager.register(MainConfig)

    parser = argparse.ArgumentParser(description='Bandit and Bayesian Optimization Experimental Framework.')
    parser.add_argument("task", type=parse_task)
    parser.add_argument("experiment", nargs='?')
    parser.add_argument("--save", help='Path where to save to.')
    parser.add_argument("--config", required=False, default=None)
    parser.add_argument("--include", required=False, default=None, type=str, nargs='+')
    parser.add_argument("--overwrite", required=False, action='store_true')
    parser.add_argument("--plots", required=False, help='Immediately plot after run.', nargs='*')
    parser.add_argument("--aggregator", required=False, help='Immediately plot after run.', nargs='*')
    parser.add_argument("--remote", required=False, action='store_true', help='Run remotely using ceramo.')
    parser.add_argument("-m", "--message", required=False, help='Message.')

    args = parser.parse_args()
    initialize_framework()

    args.task(args)


@celeryd_after_setup.connect
def initialize_framework(default_config='default.yaml', **kwargs):
    """
    Reads default config if exists
    Args:
        config_data:

    Returns:

    """

    if os.path.exists(default_config):
        config_manager.load_yaml(default_config, section=MainConfig._section)
        config_manager.update_config(main_config)
        register_modules(main_config.modules)

        # set logging levels
        get_logger.set_level(main_config.log_level_console)
        get_logger.set_filehandler_level(main_config.log_level_file)

        if main_config.plotting_backend != None:
            import matplotlib.pyplot as plt
            plt.switch_backend(main_config.plotting_backend)

        # reload config, now that we have registered the modules.
        config_manager.load_yaml(default_config)
    else:
        logger.warning(f'Did not find default config ("{default_config}").')

def register_modules(modules):
    for m in modules:
        if isinstance(m, str):
            m = locate(m)
        config_manager.register(m)
        logger.info(f'Registered module "{m.__name__}".')

def create_experiment(name, overwrite=False):
    experiment = main_config.experiment(main_config.experiment_dir)  # create experiment instance

    # if exists and overwrite is True, delete the directory
    if experiment.exists(name):
        if overwrite:
            logger.info(f'Deleting existing experiment in "{experiment.path(name)}"')
            shutil.rmtree(experiment.path(name))
        else:
            logger.warning(f'Experiment directory "{experiment.path(name)}" already exists.')

    # check if no experiment with this name has been created before
    if not experiment.exists(name):
        experiment.create(name)

    return experiment

def create(args):
    """
    febo create
    """
    if args.config is not None:
        load_config(args.config)

    experiment = main_config.experiment(main_config.experiment_dir)

    # if experiment directory exists and overwrite flag is set, ask to delete it
    if args.experiment is not None and args.overwrite and experiment.exists(args.experiment):
        if utils.query_yes_no(f"The experiment {experiment.path(args.experiment)} already exists. Do you want to delete the directory (all data lost)?", default="no"):
            shutil.rmtree(experiment.path(args.experiment))
        else:
            exit()

    experiment.create(args.experiment)

def run(args):
    """
    febo run
    """
    # load config to get experiment class
    path = os.path.join(main_config.experiment_dir, args.experiment)
    load_config(os.path.join(path, 'experiment.yaml'))

    experiment = main_config.experiment(main_config.experiment_dir)
    try:
        experiment.load(args.experiment)
        experiment.start(remote=args.remote)

        # if --plots is set, also call plot action
        if not args.plots is None and len(args.plots) > 0:
            plot(args, experiment)
    finally:
        experiment.close()

def sync(args):
    path = os.path.join(main_config.experiment_dir, args.experiment)
    load_config(os.path.join(path, 'experiment.yaml'))

    experiment = main_config.experiment(main_config.experiment_dir)
    try:
        experiment.load(args.experiment)
        experiment.sync(main_config.sync_dir)
    finally:
        experiment.close()

def plot(args, experiment=None):
    """
    febo plot
    """

    if experiment is None:
        path = os.path.join(main_config.experiment_dir, args.experiment)
        load_config(os.path.join(path, 'experiment.yaml'))

        experiment = main_config.experiment(main_config.experiment_dir)
        experiment.load(args.experiment)

    if args.plots is None or len(args.plots) == 0:
        logger.warning('No plots specified. Use "--plots path.to.PlottingClass1 path.to.PlottingClass2" to specify plots.')
    for plot_class in args.plots:
        plot_class = plot_class.split(':')
        plot = None
        try:
            plot = locate(plot_class[0])
        except:
            logger.error(f'Could not locate {plot_class}')
        if plot is not None:
            plot = plot(experiment)
            group_id = None
            if len(plot_class) > 1 and plot_class[1]:
                group_id = int(plot_class[1])
            run_id = None
            if len(plot_class) > 2 and plot_class[2]:
                run_id = int(plot_class[2])

            plot.plot(show=True, group_id=group_id, run_id=run_id)


def aggregate(args, experiment=None):
    """
    febo aggregate
    """

    if experiment is None:
        path = os.path.join(main_config.experiment_dir, args.experiment)
        load_config(os.path.join(path, 'experiment.yaml'))

        experiment = main_config.experiment(main_config.experiment_dir)
        experiment.load(args.experiment)

    for cls in args.aggregator:
        aggr = locate(cls)
        aggr(experiment)

def note(args):
    path = os.path.join(main_config.experiment_dir, args.experiment, 'notes.txt')

    with open(path, "a") as notes:
        note = f"{get_timestamp()}: {args.message} \n"
        notes.write(note)
        logger.info(f"added note: {note}")


def doc(args):
    """
    febo doc (convenience function to open documentation in browser, only to be kept during development.)
    """
    import webbrowser
    webbrowser.open_new_tab(os.path.abspath(os.path.join(os.path.dirname(__file__), '../doc/_build/html/index.html')))

def unknown(args):
    """
    catching unknown command line arguments.
    """
    print(f"Invalid command line arguments.")

def list_config(args):
    """
    febo config
    """



    # # add cwd to path
    # cwd = os.getcwd()
    # sys.path.insert(0, cwd)

    # print config
    print(config_manager.get_yaml(include_default=True))
    if args.save:
        config_manager.write_yaml(args.save, include_default=True)
        print(f"Saved config to {args.save}.")

arg_choices = [create, run, sync, plot, aggregate, ('config', list_config), doc, note]


def parse_task(astring):
    for task in arg_choices:
        if hasattr(task, '__call__') and task.__name__ == astring:
            return task
        if isinstance(task, tuple) and task[0] == astring:
            return task[1]

    raise argparse.ArgumentTypeError(f'Invalid argument "{astring}".')


def load_config(path):
    config_manager.load_yaml(path)
    config_manager.update_config(main_config)


if __name__ == "__main__":
     # import all submodules to generate all config instances
    main()