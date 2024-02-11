import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import commons as cm
from utils import opt_utils as ou
from utils import visulaization as vis

import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
from pymoo.core.callback import Callback
import pandas as pd


class TBLogger(Callback):
    """
            Tensorboard Logger callback
    """

    def __init__(self, obj_names, obj_weights, constr_names,
                 minimization=True, combine_multi_obj=True,
                 save_history=True, result_save_freq=1,
                 log_root_dir="./tb_logs/", tag=None):
        """
        Tensorboard logger for optimization
        :param obj_names: list of objective names
        :param obj_weights: np.array of objective weights
        :param constr_names: list of contraint names
        :param minimization: whether it is a minimization
        :param combine_multi_obj: whether to combine multiple objectives for a single objective
        :param result_save_freq: how frequently the optimal solution is saved
        :param log_root_dir: where to save the log files
        :param tag: tag of the log, if none then run time is used
        """
        super().__init__()
        self.tag = tag if tag is not None else cm.get_current_time_str()
        self.log_dir = f"{log_root_dir}{self.tag}"
        self.tb_logger = SummaryWriter(self.log_dir)

        self.obj_names = obj_names
        self.obj_weights = np.array(obj_weights).flatten().reshape(1, -1)
        self.n_objs = len(obj_names)
        self.combined_reward = combine_multi_obj  # whether to combine multi-objs to a single reward

        self.constr_names = constr_names
        if constr_names is not None:
            self.n_constr = len(constr_names)
        else:
            self.n_constr = 0

        self.minimization = minimization

        self.best_ever_X = None
        self.best_ever_result = np.array(
            [sys.float_info.max if minimization else -sys.float_info.max, ] * self.n_objs
        )
        self.result_save_freq = result_save_freq
        self.save_history = save_history
        self.history = []

    def notify(self, algorithm=None, **kwargs):
        if algorithm is None:
            batch_objs = kwargs.get("batch_objs", None)
            batch_constrs = kwargs.get("bacth_constraints", None)
            batch_X = kwargs.get("batch_X", None)
            n_step = kwargs.get("n_step", None)
        else:
            # pymoo compatibility
            batch_objs = algorithm.pop.get("F")
            batch_constrs = algorithm.pop.get("G")
            batch_X = algorithm.pop.get("X")
            n_step = algorithm.n_gen

        assert batch_objs is not None
        assert batch_X is not None
        assert n_step is not None

        batch_objs = batch_objs.reshape(-1, self.n_objs)
        if not self.minimization:  # the reward has been multiplied with -1.0 because the optimizer always minimize
            batch_objs = batch_objs * -1.0

        if self.save_history:
            self.history.append((n_step, batch_X, batch_objs))
            cm.serialize_obj(self.history, os.path.join(self.log_dir, f"history.pkl"))

        # log the opt value ever seen in the batch
        # (NOTE the optimum is calculated individually for each obj)
        for i in range(self.n_objs):
            batch_opt_obj_i = np.nanmin(batch_objs[:, i]) if self.minimization \
                else np.nanmax(batch_objs[:, i])
            self.tb_logger.add_scalar(tag=f"batch_individual_opt/{self.obj_names[i]}",
                                      scalar_value=batch_opt_obj_i,
                                      global_step=n_step)

        # log the opt constraints ever seen in the batch
        # (NOTE the optimum is calculated individually for each constr.)
        if batch_constrs is not None:
            batch_constrs = np.atleast_2d(batch_constrs)
            for i in range(self.n_constr):
                batch_opt_constr_i = np.nanmin(batch_constrs[:, i])
                self.tb_logger.add_scalar(tag=f"batch_individual_constr/{self.constr_names[i]}",
                                          scalar_value=batch_opt_constr_i,
                                          global_step=n_step)

        # update the best-ever
        best_updated = False
        batch_opt_ind = None  # the optimal point in current batch
        if (self.n_objs == 1) or (self.n_objs > 1 and self.combined_reward is True):
            # -- single-obj --
            if self.combined_reward:
                batch_scores = (batch_objs * self.obj_weights).sum(axis=1).flatten()
            else:
                batch_scores = batch_objs.flatten()
            batch_opt_ind = np.nanargmin(batch_scores) if self.minimization \
                else np.nanargmax(batch_scores)  # select the optimum if it is single obj.
            batch_opt_score = batch_scores[batch_opt_ind]

            # Is it better than the previous best?
            prev_opt_score = (self.best_ever_result * self.obj_weights).sum()
            if ((self.minimization is True) and (batch_opt_score < prev_opt_score)) \
                    or ((self.minimization is False) and (batch_opt_score > prev_opt_score)):
                best_updated = True
                self.best_ever_result = batch_objs[batch_opt_ind]
                self.best_ever_X = batch_X.iloc[batch_opt_ind] \
                    if isinstance(batch_X, pd.DataFrame) else batch_X[batch_opt_ind]
        else:
            # -- multi-obj --
            # find the pareto-frontier
            cur_pareto_frontier = self.best_ever_result
            cur_pareto_frontier_ind = None
            for i in range(batch_objs.shape[0]):
                sol_i = batch_objs[i, :]
                if ou.dominate(sol_i, cur_pareto_frontier, self.minimization):
                    # got a better frontier
                    best_updated = True
                    cur_pareto_frontier = batch_objs[i, :]
                    cur_pareto_frontier_ind = i

            # find the non-dominate set w.r.t. current frontier
            non_dominate_inds = []
            if best_updated:
                non_dominate_inds.append(cur_pareto_frontier_ind)
                for i in range(batch_objs.shape[0]):
                    sol_i = batch_objs[i, :]
                    if ou.non_dominate(sol_i, cur_pareto_frontier, self.minimization):
                        non_dominate_inds.append(i)

            # find the batch_opt and update best ever
            batch_scores = (batch_objs * self.obj_weights).sum(axis=1)
            if best_updated:
                non_dom_scores = batch_scores[non_dominate_inds]
                sorted_nds = sorted(zip(non_dominate_inds, non_dom_scores),
                                    key=lambda x: x[1],
                                    reverse=True)
                batch_opt_ind = sorted_nds[-1][0] if self.minimization else sorted_nds[0][0]

                # update best ever
                self.best_ever_result = batch_objs[batch_opt_ind, :]
                if isinstance(self.best_ever_X, pd.DataFrame):
                    self.best_ever_X = batch_X.iloc[batch_opt_ind, :] \
                        if isinstance(batch_X, pd.DataFrame) else batch_X[batch_opt_ind, :]
                else:
                    self.best_ever_X = batch_X[batch_opt_ind, :]
            else:
                batch_opt_ind = np.nanargmin(batch_scores) if self.minimization \
                    else np.nanargmax(batch_scores)

        # log batch opt objs in terms of obj weights
        for i in range(self.n_objs):
            self.tb_logger.add_scalar(tag=f"batch_opt/{self.obj_names[i]}",
                                      scalar_value=batch_objs[batch_opt_ind, i],
                                      global_step=n_step)

        # log the best-ever
        for i in range(self.n_objs):
            self.tb_logger.add_scalar(tag=f"best-ever/{self.obj_names[i]}",
                                      scalar_value=self.best_ever_result[i],
                                      global_step=n_step)

        # save the results
        if n_step % self.result_save_freq == 0:
            cm.serialize_obj(
                batch_X.iloc[batch_opt_ind] if isinstance(batch_X, pd.DataFrame) \
                    else batch_X[batch_opt_ind],
                os.path.join(self.log_dir, f"batch_opt_X_{n_step}.pkl"))
            cm.serialize_obj(batch_objs[batch_opt_ind],
                             os.path.join(self.log_dir, f"batch_opt_objs_{n_step}.pkl"))
            cm.serialize_obj(self.best_ever_X,
                             os.path.join(self.log_dir, f"best_X_{n_step}.pkl"))
            cm.serialize_obj(self.best_ever_result,
                             os.path.join(self.log_dir, f"best_result_{n_step}.pkl"))

        return best_updated, batch_opt_ind, self.best_ever_X, self.best_ever_result


class OptLogger(TBLogger):
    def notify(self, algorithm=None, **kwargs):
        n_step = kwargs.get("n_step", None)
        best_updated, \
        batch_opt_ind, best_ever_X, best_ever_result = super().notify(algorithm, **kwargs)

        batch_X = kwargs.get("batch_X", None)

        # plot history
        h_fig = vis.visualize_opt_history_y(self.history, self.obj_names)
        if h_fig is not None:
            self.tb_logger.add_figure(f"opt_history", h_fig, n_step)

        if batch_X is not None:
            x_fig = vis.visualize_opt_history_x(self.history)
            self.tb_logger.add_figure('batch_x', x_fig, global_step=n_step, close=True)

        return best_updated, batch_opt_ind, best_ever_X, best_ever_result
