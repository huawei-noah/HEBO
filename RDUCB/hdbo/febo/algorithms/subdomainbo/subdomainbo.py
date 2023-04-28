from febo.algorithms import Algorithm, AlgorithmConfig, ModelMixin
from febo.utils import get_logger, locate
from febo.utils.config import assign_config, ConfigField
from febo.solvers import GridSolver, CandidateSolver, FiniteDomainSolver
import numpy as np
import os
import matplotlib.pyplot as plt
from febo.algorithms.subdomainbo.utils import maximize, dimension_setting_helper
from .utils import plot_parameter_changes, plot_model_changes
from .subdomain import LineDomain, TrustRegionDomain
from scipy.spatial.distance import pdist

logger = get_logger('algorithm')


class SubDomainBOConfig(AlgorithmConfig):
    points_in_max_interval_to_stop = ConfigField(10)
    min_queries_line = ConfigField(5)
    max_queries_line = ConfigField(10)
    min_queries_tr = ConfigField('1*d')
    max_queries_tr = ConfigField('1*d')
    tr_radius = ConfigField(0.1)
    tr_method = ConfigField('grad')
    line_boundary_margin = ConfigField(0.05)
    plot = ConfigField(False)
    plot_every_step = ConfigField(False)

    acquisition = ConfigField('febo.algorithms.subdomainbo.acquisition.ucb')
    _section = 'algorithm.subdomainbo'

def sample_grad_gp(model, x0, scale, eps=0.01):
    points = x0 + np.eye(len(x0))*scale*eps
    points = np.vstack((x0, points))
    Y = model.gp.posterior_samples_f(points, size=1).flatten()
    return (Y[1:] - Y[0])/(scale*eps)

def mean_grad_gp(model, x0, scale, eps=0.01):
    return model.gp.predictive_gradients(x0.reshape(1,-1))[0].flatten()
    # points = x0 + np.eye(len(x0))*scale*eps
    # np.hstack((x0, points))
    # Y = model.mean(points).flatten()
    # return (Y[1:] - Y[0])/(scale*eps)


@assign_config(SubDomainBOConfig)
class SubDomainBO(ModelMixin, Algorithm):
    """
    This class is used to run a 1-dim version of BO on a Sumdomain
    """

    def initialize(self, **kwargs):
        super(SubDomainBO, self).initialize(**kwargs)

        self._best_x = self.x0.copy()

        self._best_direction = None
        self._phase = 'best'
        self._iteration = 0

        self._parameter_names = kwargs.get('parameter_names')

        self._max_queries_line = dimension_setting_helper(self.config.max_queries_line, self.domain.d)
        self._min_queries_line = dimension_setting_helper(self.config.min_queries_line, self.domain.d)
        self._max_queries_tr = dimension_setting_helper(self.config.max_queries_tr, self.domain.d)
        self._minx_queries_tr = dimension_setting_helper(self.config.min_queries_tr, self.domain.d)
        self._point_type_addition = ''

        self.__acquisition = locate(self.config.acquisition)

        # init global models
        # handled by ModelMixin so far

    def _next(self, context=None):
        additional_data = {'iteration' : self._iteration}
        # sampling phases
        if self._phase == 'best':
            additional_data['point_type'] = 'best'
            return self._best_x, additional_data

        if self._phase == 'line':
            additional_data['point_type'] = 'line'  + self._point_type_addition
            self._point_type_addition = ''
            x_next = self._line_solver_step()
            # if a tuple is returned, it contains x,m
            if isinstance(x_next, tuple):
                x_next, m = x_next
                additional_data['m'] = m
                logger.info(f"Choosing {m} measurements.")

            return x_next, additional_data

        if self._phase == 'tr':
            additional_data['point_type'] = 'tr' + self._point_type_addition
            self._point_type_addition = ''
            x_next = self._tr_solver_step()
            if isinstance(x_next, tuple):
                x_next, m = x_next
                additional_data['m'] = m
                logger.info(f"Choosing {m} measurements.")
            return x_next, additional_data

    def add_data(self, data):
        super().add_data(data)

        # evaluate stopping conditions
        if self._phase == 'line':
            # add line data
            self._line_add_data(data)
            self._best_x = self._line_solver_best()
            self._best_x_list.append(self._best_x.copy())

            if self._line_solver_stop():
                self._line_solver_finalize()
                self._phase = 'best'

        elif self._phase == 'tr':
            # add tr data
            self._tr_add_data(data)
            self._best_x = self._tr_solver_best()

            if self._tr_solver_stop():
                # compute best direction
                self._best_direction = self._tr_solver_best_direction()
                self._tr_solver_finalize()
                self._phase = 'best'

        elif self._phase =='best':
            self._iteration += 1

            self._phase, subdomain = self._get_new_subdomain()
            logger.info(f'best_x evaluate, y={data["y"]}')


            if self._phase == 'line':
                self._line_solver_init(subdomain)
            elif self._phase == 'tr':
                self._tr_solver_init(subdomain)

            logger.info(f'starting {self._iteration}, {self._phase}-solver.')

    def _get_new_subdomain(self):
        if self._iteration % 2 == 0:
            return 'tr',
        else:
            return 'line',

    def _line_solver_init(self, line_domain):
        self._line_data = []
        self._best_x_list = []
        self._line_domain = line_domain
        self._line_solver = GridSolver(domain=line_domain, points_per_dimension=300)

    def _line_add_data(self, data):
        self._line_data.append(data)

    def _line_max_ucb(self):
        max_ucb = self._line_solver.minimize(lambda X : -self.model.ucb(self._line_domain.embed_in_domain(X)))[1]
        return -max_ucb

    def _line_solver_stop(self):
        # don't stop below min-queries
        if len(self._line_data) <= self._min_queries_line:
            return False

        # accuracy of maximum < 1%
        if self._line_max_ucb() - self.model.lcb(self._best_x) < 0.01*self.model.mean(self._best_x):
            logger.warning("Uncertainty at best_x reduced to 1%, stopping line.")
            return True

        # best_x didn't change after half the samples
        # flexible_query_range = max(self._max_queries_line - self._min_queries_line, 6)
        # if len(self._line_data) >= self._min_queries_line + flexible_query_range/4:
        #     # maximum distance of last few best_x did not change by more than 2 % on domain
        #     if np.max(pdist(self._best_x_list[-flexible_query_range // 2:], w=1/self.domain.range**2)) < 0.02:
        #         logger.warning(f"No best_x change in {flexible_query_range // 2} steps. Stopping line.")
        #         return True

        # stop at max queries
        return len(self._line_data) >= self._max_queries_line

    def _line_solver_best(self):
        boundary_margin = self._line_domain.range * self.config.line_boundary_margin
        def mean(X):
            # return model mean on line, but ignore a margin at the boundary to account for boundary effects of the gp
            return -self.model.mean(self._line_domain.embed_in_domain(X)) \
                   + 10e10*np.logical_or(X < self._line_domain.l + boundary_margin, X > self._line_domain.u - boundary_margin)
        x_line = self._line_solver.minimize(mean)[0]
        return self._line_domain.embed_in_domain(x_line).flatten()

    def _line_solver_step(self):
        if self.config.plot_every_step:
            self._save_line_plot(with_step_num=True)

        x_line = self._line_solver.minimize(lambda X : -self.global_acquisition(self._line_domain.embed_in_domain(X)))[0]
        return self._line_domain.embed_in_domain(x_line).flatten()

    def _line_solver_finalize(self):
        if self.config.plot:
            self._save_line_plot()

    def _save_line_plot(self, with_step_num=False):
        """
        Save a plot of the current line. The plot is generated in .plot_line(...)
        """
        f = plt.figure()
        axis = f.gca()

        self.plot_line(axis=axis)

        # save plot
        group_id = self.experiment_info.get("group_id", "")
        if group_id is None: # group_id might be set to None already in self.experiment_dir
            group_id = ""

        path = os.path.join(self.experiment_info["experiment_dir"], "plots", str(group_id), str(self.experiment_info.get("run_id", "")))
        os.makedirs(path, exist_ok=True)
        f.subplots_adjust(top=0.71)
        if with_step_num:
            path = os.path.join(path, f'Iteration_{self._iteration}_{self.t}.pdf')
        else:
            path = os.path.join(path, f'Iteration_{self._iteration}.pdf')
        f.savefig(path)
        logger.info(f'Saved line plot to {path}')
        plt.close()

    def plot_line(self, axis, steps=300):
        """
        This function uses the datapoints measured in one dim and plots these together with the standard deviation
        and mean of the model to check the lengthscale. It returns the plots into a folder with one plot per line in the dropout algorithm

        :param axis: axis to plot on
        :param line_data:
        :param steps:
        """

        # first create evaluation grid with correct bounds on the sub-domain
        X_eval = np.linspace(self._line_domain.l[0], self._line_domain.u[0], steps)

        # then we evaluate the mean and the variance by projecting back to high-d space
        X_eval_embedded = self._line_domain.embed_in_domain(X_eval.reshape(-1, 1))
        mean, var = self.model.mean_var(X_eval_embedded)
        mean, std = mean.flatten(), np.sqrt(var).flatten()

        # we plot the mean, mean +/- std and the data points
        axis.fill_between(X_eval, mean - std, mean + std, alpha=0.4, facecolor='grey', color='C0')
        axis.plot(X_eval, mean, color='C0')


        data_x = [self._line_domain.project_on_line(p['x']).flatten() for p in self._line_data]
        data_y = [p['y'] for p in self._line_data]
        axis.scatter(data_x, data_y,marker='x', c='C0')

        # starting and best_predicted point
        axis.axvline(self._line_domain.project_on_line(self._line_domain.x0), color='C0', linestyle='--')
        axis.axvline(self._line_domain.project_on_line(self._best_x), color='C0')

        # add some information in the title
        axis.set_title(f'Iteration: {self._iteration}'
                       f'\nbeta= {round(self.model.beta,3)}, variance= {round(self.model.gp.kern.variance[0],3)}, '
                       f'\nnoise variance= {round(self.model.gp.Gaussian_noise.variance[0],5)}')

        return X_eval, X_eval_embedded, data_x

    def _tr_solver_init(self, tr_domain):
        self._tr_domain = tr_domain
        self._tr_solver = FiniteDomainSolver(tr_domain)
        self._tr_data = []

    def _tr_add_data(self, data):
        self._tr_data.append(data)

    def _tr_solver_step(self):
        if self.config.tr_method == 'ball':
            return self._tr_solver.minimize(lambda X : -self.global_acquisition(X))[0]
            # print(self._tr_domain.radius)
            # print(np.sum((next_x - self._tr_domain.x0)**2/self._tr_domain.radius**2))
            # return next_x
        if self.config.tr_method == 'grad':
            grad_sample = sample_grad_gp(self.model, self._tr_domain.x0, self._tr_domain.radius, 0.01)
            # print('grad', grad_sample)
            # print('x0', self._tr_domain.x0)
            # normalize gradient
            grad_sample /= np.linalg.norm(grad_sample)
            # scale gradient to boundary of TR
            next_x = self.domain.project(self._tr_domain.x0 + grad_sample*self._tr_domain.radius)
            # print('next_x)
            return next_x

    def _tr_solver_best(self):
        # return self._tr_solver.minimize(lambda X : -self.model.mean(X))[0]

        if self.config.tr_method == 'ball':
            return self._tr_solver.minimize(lambda X : -self.model.mean(X))[0]
        if self.config.tr_method == 'grad':
            grad = mean_grad_gp(self.model, self._tr_domain.x0, self._tr_domain.radius, 0.001)
            tr_best = self._tr_domain.radius*grad/np.linalg.norm(grad)

            # line search on model mean
            candidates = self.domain.project(self._tr_domain.x0 + np.outer(np.linspace(0,1,100), tr_best.reshape(1,-1)))
            return maximize(self.model.mean, candidates)[0]


    def _tr_solver_best_direction(self):
        direction = (self._best_x - self._tr_domain.x0).reshape(1, -1)
        # if self.config.tr_method == 'ball':
        #     direction = (self._best_x - self._tr_domain.x0).reshape(1, -1)
        # if self.config.tr_method == 'grad':
        #     direction = mean_grad_gp(self.model, self._tr_domain.x0, self._tr_domain.radius, 0.001)

        # if change is less  than 2% of tr-radius or increase is less then 0.5%, pick a random direction
        if np.linalg.norm(direction/self._tr_domain.radius) < 0.02:
            logger.warning('change in best_x < 2% of trust-region, picking random direction.')
            direction = self.get_random_direction()
        else:
            y_x0 = self.model.mean(self._tr_domain.x0)
            y_new = self.model.mean(self._best_x)
            if y_new/y_x0 < 1.005:
                logger.warning('predicted objective increase at best_x < 0.5%, picking random direction.')
                direction = self.get_random_direction()

        return direction

    def _tr_solver_stop(self):
        return len(self._tr_data) >= self._max_queries_tr

    def _tr_solver_finalize(self):
        if self.config.plot:
            self._tr_save_plot()

    def _tr_save_plot(self, with_step_num=False):
        """
              Save a plot of the current line. The plot is generated in .plot_line(...)
              """
        fig, axis = plt.subplots(ncols=1, figsize=(self.domain.d, 4))



        plot_parameter_changes(axis, self._parameter_names, self._tr_domain.x0, self._best_x, self.domain.l, self.domain.u, self._tr_domain.radius, self.x0)

        x0 = self._tr_domain.x0
        xnew = self._best_x
        y_x0 = np.asscalar(self.model.mean(x0.reshape(1, -1)))
        y_xnew = np.asscalar(self.model.mean(xnew.reshape(1,-1)))
        # ucb_xnew = np.asscalar(self.model.ucb(xnew.reshape(1,-1)))
        std_xnew = np.asscalar(self.model.std(xnew.reshape(1,-1)))

        y_coord = np.empty(self.domain.d)
        # ucb_coord = np.empty(self.domain.d)
        for i in range(self.domain.d):
            axis_points = self._tr_domain.get_axis_points(i)
            y_coord[i] = np.max(self.model.mean(axis_points))
            # ucb_coord[i] = np.max(self.model.ucb(axis_points))

        plot_model_changes(axis, y_x0, y_xnew, std_xnew, y_coord)

        # save plot
        group_id = self.experiment_info.get("group_id", "")
        if group_id is None:  # group_id might be set to None already in self.experiment_dir
            group_id = ""

        path = os.path.join(self.experiment_info["experiment_dir"], "plots", str(group_id),
                            str(self.experiment_info.get("run_id", "")))
        os.makedirs(path, exist_ok=True)
        # fig.subplots_adjust(wspace=0.4)
        if with_step_num:
            path = os.path.join(path, f'Iteration_{self._iteration}_{self.t}.pdf')
        else:
            path = os.path.join(path, f'Iteration_{self._iteration}.pdf')
        fig.savefig(path, bbox_inches="tight")
        logger.info(f'Saved trust-region plot to {path}')
        plt.close()

    def global_acquisition(self, X):
        return self.__acquisition(self.model, X)

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        fields += [('iteration', 'i')]
        fields += [('direction', '(1,%s)f' % self.domain.d)]
        fields += [('point_type', 'S25')]
        return fields

    def best_predicted(self):
        return self.domain.project(self._best_x)

    def get_random_direction(self):
        """
        creates a random directional vector in d = domain.d dimensions
        :return: return a vector in shape (1, self.domain.d)
        """
        direction = np.random.normal(size=self.domain.d).reshape(1,-1)
        direction /= np.linalg.norm(direction)
        direction *= self.domain.range  # scale direction with parameter ranges, such that the expected change in each direction has the same relative magnitude
        return direction


class CoordinateLineBO(SubDomainBO):
    """
    Bayesian optimization along the coordinates.
    """

    def plot_line(self, axis, steps=300):
        info = super().plot_line(axis, steps=steps)
        # set the parameter name as x-label
        if self._parameter_names is None:
            axis.set_xlabel(f'x_{self._iteration % self.domain.d}')
        else:
            axis.set_xlabel(self._parameter_names[self._iteration % self.domain.d])
        return info

    def _get_new_subdomain(self):
        direction = np.eye(self.domain.d)[self._iteration % self.domain.d].reshape(1,-1)
        line_domain = LineDomain(self.domain, self._best_x, direction)
        return 'line', line_domain


class RandomLineBO(SubDomainBO):
    """
    Bayesian optimization in random directions.
    """

    def _get_new_subdomain(self):
        direction = self.get_random_direction()
        line_domain = LineDomain(self.domain, self._best_x.copy(), direction)
        return 'line', line_domain


class AscentLineBO(SubDomainBO):
    """
    Bayesian Optimization with alternateting trust-region and line-search.
    """

    def _get_new_subdomain(self):
        radius = self.domain.range * self.config.tr_radius
        if self._iteration % 2 == 1:
            return 'tr', TrustRegionDomain(self.domain, self._best_x.copy(), radius=radius)
        else:
            return 'line', LineDomain(self.domain, self._best_x.copy(), self._best_direction)



