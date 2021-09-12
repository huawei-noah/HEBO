import os
from typing import Dict, Optional, Any, List

import numpy as np

from utils.utils_save import ROOT_PROJECT
from weighted_retraining.weighted_retraining.expr.expr_dataset import get_filepath
from weighted_retraining.weighted_retraining.robust_opt_scripts.robust_opt_expr import get_root_path
from weighted_retraining.weighted_retraining.utils import RegretPlotter

EXPR_DIR = os.path.join(ROOT_PROJECT, 'weighted_retraining', 'data', 'expr')


class RobustExprRegretPlotter(RegretPlotter):

    def __init__(self, budget: int, lso_strategy: str, weight_type, k: float, r: int, predict_target: bool,
                 target_predictor_hdims: List[int], metric_loss: str, metric_loss_kw: Dict[str, Any], acq_func_id: str,
                 covar_name: str, acq_func_kwargs: Dict[str, Any], input_wp: bool, output_wp: bool,
                 random_search_type: Optional[str],
                 dataset_seed: int, ignore_percentile, good_percentile,  n_max_epochs: int,
                 estimate_rec_error: bool, cost_aware_alpha_sched: Optional[str],
                 mis_preprocessing: bool):
        """
        Args:
            lso_strategy: type of optimisation
            weight_type: type of weighting used for retraining
            k: weighting parameter
            r: period of retraining
            predict_target: whether generative model also predicts target value
            target_predictor_hdims: latent dims of target MLP predictor
            metric_loss: metric loss used to structure embedding space
            metric_loss_kw: kwargs for metric loss
            acq_func_id: name of acquisition function
            acq_func_kwargs: acquisition function kwargs
            covar_name: name of kernel used for the GP
            input_wp: whether input warping is used (Kumaraswarmy)
            output_wp: whether output warping is used
            random_search_type: random search specific strategy
            dataset_seed: seed that has been used to generate the dataset
            budget: number of acquisition steps to show
            ignore_percentile: portion of original equation dataset ignored
            good_percentile: portion of good original equation dataset included
            n_max_epochs: max number of epochs on which model has been trained
            estimate_rec_error:  Whether to estimate reconstruction error when new points are acquired
            cost_aware_alpha_sched: schedule for cost-aware acquisition function parameter `alpha`

        """
        super().__init__(budget=budget,
                         lso_strategy=lso_strategy, weight_type=weight_type, k=k, r=r, predict_target=predict_target,
                         target_predictor_hdims=target_predictor_hdims, metric_loss=metric_loss,
                         metric_loss_kw=metric_loss_kw, acq_func_id=acq_func_id, covar_name=covar_name,
                         input_wp=input_wp, output_wp=output_wp, random_search_type=random_search_type,
                         maximise=False)
        self.dataset_seed = dataset_seed
        self.ignore_percentile = ignore_percentile
        self.good_percentile = good_percentile
        self.n_max_epochs = n_max_epochs
        self.estimate_rec_error = estimate_rec_error
        self.cost_aware_alpha_sched = cost_aware_alpha_sched
        self.acq_func_kwargs = acq_func_kwargs
        self.mis_preprocessing = mis_preprocessing

    def get_expr_start_score(self):
        return get_expr_start_score(
            ignore_percentile=self.ignore_percentile,
            good_percentile=self.good_percentile,
            save_dir=EXPR_DIR,
            dataseed=self.dataset_seed
        )

    def get_root_path(self):
        return get_root_path(
            lso_strategy=self.lso_strategy,
            weight_type=self.weight_type,
            k=self.k,
            r=self.r,
            ignore_percentile=self.ignore_percentile,
            good_percentile=self.good_percentile,
            predict_target=self.predict_target,
            hdims=self.target_predictor_hdims,
            metric_loss=self.metric_loss,
            metric_loss_kw=self.metric_loss_kw,
            acq_func_id=self.acq_func_id,
            acq_func_kwargs=self.acq_func_kwargs,
            covar_name=self.covar_name,
            input_wp=self.input_wp,
            random_search_type=self.random_search_type,
            n_max_epochs=self.n_max_epochs,
            estimate_rec_error=self.estimate_rec_error,
            cost_aware_alpha_sched=self.cost_aware_alpha_sched,
            mis_preprocessing=self.mis_preprocessing
        )


def get_expr_start_score(ignore_percentile: float, good_percentile: float, save_dir: str, dataseed: int):
    dataset_path = get_filepath(
        ignore_percentile=ignore_percentile,
        save_dir=save_dir,
        seed=dataseed,
        good_percentile=good_percentile
    )
    with np.load(dataset_path) as npz:
        all_properties = npz['scores']
    return np.min(all_properties)
