from febo.algorithms.model import ModelMixin
from febo.algorithms.safety import SafetyMixin
from febo.controller.simple import SimpleControllerConfig
from febo.models.gpy import GP
from febo.controller import SimpleController
from febo.utils import locate, get_logger

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Plotting library could not be loaded.")
import numpy as np
from febo.plots.utilities import plot_contour_gp, plot_x_and_f, plot_safeset
from febo.utils.config import ConfigField, assign_config, config_manager


logger = get_logger("controller")

class PlottingControllerConfig:
    plots = ConfigField([])
    _section = 'controller'

@assign_config(PlottingControllerConfig)
class PlottingMixin:

    def finalize(self, *args, **kwargs):
        res = super().finalize(*args, **kwargs)
        for plot_cls in self.config.plots:
            try:
                plot = locate(plot_cls)
                plot = plot(self.experiment)
                plot.plot(show=False, group_id=self.group_id, run_id=self.run_id)
            except Exception as e:
                logger.error(f'Error {str(e)} while plotting {plot_cls}')
                logger.info(f"Here is the stack trace:\n{traceback.format_exc()} ")

        return res


@assign_config(PlottingControllerConfig)
class PlottingController(PlottingMixin, SimpleController):
    """ Controller for testing purposes. Has options for plotting. """

    pass

    # def _plot_contour(self):
    #     if self.plot_safety_constraints:
    #         num_subplots = self.algorithm.num_safety_constraints + 1
    #         if self.algorithm.lower_bound_objective_value is not None:
    #             num_subplots -= 1
    #
    #         f, axes = plt.subplots(1, num_subplots, sharey=True)
    #     else:
    #         axis = plt.gca()
    #         axes = [axis, ]
    #
    #
    #     l = self.environment.domain.l
    #     u = self.environment.domain.u
    #     linspaces = [np.linspace(l[0], u[0], 20, ), np.linspace(l[1], u[1], 20, )]
    #     if isinstance(self.algorithm, ModelMixin):
    #
    #         plot_contour_gp(self.algorithm.model,linspaces ,axis=axes[0])#, green_points=self.algorithm.safe_points)
    #         # red_points=self.algorithm.potential_maximizers,
    #         # blue_points=self.algorithm.potential_expanders)
    #     else:
    #         plot_x_and_f(lambda x: x, linspaces, self.dset["x"],
    #                      axis=axes[0])
    #
    #     if isinstance(self.algorithm, SafetyMixin) and self.config.plot_safeset:
    #         plot_safeset(self.algorithm._is_safe, linspaces, axis=axes[0])
    #         #
    #         # for i, s in enumerate(self.algorithm.s):
    #         #     if isinstance(s, GP):
    #         #         plot_contour_gp(s.gp, [np.linspace(0, 1, 20, ), np.linspace(0, 1, 20, )],
    #         #                         axis=axes[i + 1], colorbar=False, green_points=self.algorithm.safe_points)
    #         #         # red_points=self.algorithm.safe_points, green_points=self.algorithm.potential_maximizers,
    #         #         # blue_points=self.algorithm.potential_expanders)
    #     plt.show()



