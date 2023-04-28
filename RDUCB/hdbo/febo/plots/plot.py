import matplotlib.pyplot as plt

import numpy as np
from febo.utils import get_logger
from febo.utils.config import Config, ConfigField, assign_config, ClassConfigField
import scipy.stats as stats
import os

logger = get_logger("plotting")

class Plot:

    def __init__(self, experiment):
        self.experiment = experiment
        self._title = None
        self._show_legend = False

    @property
    def title(self):
        return self._title

    def plot(self, show=True, group_id=None, run_id=None):
        from febo.experiment import MultiExperiment, SimpleExperiment
        from febo.controller import SequentialController, RepetitionController

        path_subdir = ""  # is set to group_id if group_id != None
        run_id_str = None  # either "-all" if run_id == None, als "-{run_id}"

        # plot data from simple experiment, or a single data group
        if type(self.experiment) is SimpleExperiment or group_id != None:
            if group_id != None:
                path_subdir = str(self.experiment.parts[group_id].id)
                group = self.experiment.hdf5[path_subdir]
                label = self.experiment.parts[group_id].label
            else:
                group = self.experiment.hdf5
                label = self.experiment.algorithm.name

            dset_list, run_id_str = self._get_dset_list(group, run_id)

            f = plt.figure()
            axis = f.gca()
            self._plot(axis, data=dset_list, label=label)

        # plot data from MultiExperiment with no specific group_id given
        elif isinstance(self.experiment, MultiExperiment):

            # generate plots for SequentialController: each part goes into separate plot
            if issubclass(self.experiment.config.multi_controller, SequentialController):
                f, axes = plt.subplots(ncols=len(self.experiment.parts))

                for axis, item in zip(axes, self.experiment.parts):
                    group = self.experiment.hdf5[str(item.id)]
                    data_list, run_id_str = self._get_dset_list(group, run_id)
                    self._plot(axis=axis, data=data_list, label=item.label)

                    axis.set_title(item.label)

            # generate plot for RepetitionController: all parts go into same plot
            elif issubclass(self.experiment.config.multi_controller, RepetitionController):
                self._show_legend = True
                f = plt.figure()
                axis = f.gca()

                # iterate parts and plot data for each group
                for item in self.experiment.parts:
                    group = self.experiment.hdf5[str(item.id)]
                    data_list, run_id_str = self._get_dset_list(group, run_id)
                    self._plot(axis, data=data_list, label=item.label)


        if self._show_legend:
            f.legend()

        # f.legend()
        f.suptitle(self.title)

        # save figure
        path = os.path.join(self.experiment.directory, 'plots', path_subdir)
        os.makedirs(path, exist_ok=True)

        filename = self.title.lower().replace(' ', '_') + f'-{run_id_str}.pdf'
        full_path = os.path.join(path, filename)
        plt.savefig(full_path)
        logger.info(f"Saved plot to {full_path}")

        if show:
            plt.show()

    def _plot(self, axis, data, label):
        raise NotImplementedError

    def _get_dset_list(self, group, run_id):
        """
        Helper function to return list of dsets in a hdf5 group, given run_id or run_id=None
        Also return the run_id as str (this is to convert run_id=-1 to the actual run-id).
        """
        if run_id != None:
            if run_id >= 0:
                dset_name = str(run_id)
            else:
                dset_name = [*group.keys()][-1]
            return [group[dset_name]], dset_name

        else:
            return [*group.values()], "all"


class DataPlot(Plot):

    def __init__(self, experiment):
        super().__init__(experiment)
        self._plot_counter = 0

    def _plot(self, axis, data, label):
        values = []

        for dset in data:
            self._init_dset(dset)

            for t,row in enumerate(dset):
                value = self._get_value(row, t)

                if len(values) <= t:
                    values.append([])
                values[t].append(value)

        avg_values = [np.mean(row) for row in values] # need to do this row-wise because rows might have different lengths
        sterr = [stats.sem(row) for row in values]

        T = len(avg_values)


        linestyle = '-'
        if self._plot_counter > 9:
            linestyle = '--'
        axis.errorbar(range(T), avg_values, yerr=sterr, label=label, errorevery=max(1,T//20), linestyle=linestyle)
        self._plot_counter += 1

    def _init_dset(self, dset):
        pass

    def _get_value(self, row, t):
        raise NotImplementedError


class Performance(DataPlot):
    def __init__(self, experiment):
        super().__init__(experiment)
        self._title = 'Performance'


    def _get_value(self, row, t):
        if 'y_exact' in row.dtype.fields:
            return row['y_exact']
        else:
            return row['y']