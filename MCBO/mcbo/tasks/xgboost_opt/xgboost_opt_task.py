# Greatly inspired by:
# https://github.com/xingchenwan/Casmopolitan/blob/ae7f5a06206712e7776562c5c0e8f771c8780575/mixed_test_func/xgboost_hp.py
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost
from sklearn import model_selection, metrics, datasets
from sklearn.utils import Bunch

from mcbo.tasks import TaskBase


class XGBoostTask(TaskBase):
    """
    XGBoost hyperparameter tuning
    """

    @property
    def name(self) -> str:
        name = f'XGBoost Opt - {self.dataset_id}'
        if self.split != .3:
            name += f" - split {self.split}"
        return name

    def __init__(self, dataset_id: str, split: float = 0.3, split_seed: int = 0):
        """
        Args:
            dataset_id: on which dataset from sklearn to apply XGBoost (boston, mnist,...)
            split: in [0, 1], portion of the dataset used for test
            split_seed: seed used for the split
        """
        super(XGBoostTask, self).__init__()
        self.split_seed = split_seed
        self.split = split
        self.dataset_id = dataset_id
        self.data = XGBoostTask.get_data(self.dataset_id)
        self.task_type = XGBoostTask.get_task_type(self.dataset_id)

        self.original_x_bounds = np.array([[0, 1], [1, 10], [0, 10],
                                           [0.001, 1], [0, 5], ])
        self.categorical_dims = np.array([0, 1, 2])
        self.continuous_dims = np.array([3, 4, 5, 6, 7])
        self.n_vertices = np.array([2, 2, 2])

        if self.task_type == 'clf':
            stratify = self.data['target']
        else:
            stratify = None
        self.train_x, self.test_x, self.train_y, self.test_y = \
            model_selection.train_test_split(self.data['data'],
                                             self.data['target'],
                                             test_size=self.split,
                                             stratify=stratify,
                                             random_state=self.split_seed)

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:

        results = np.zeros((len(x), 1))

        for i in range(len(x)):
            results[i, 0] = self.evaluate_single_point(x.iloc[i])

        return results

    def evaluate_single_point(self, x: pd.Series) -> float:
        """
        Args
            x: hyperparams to evaluate

        Returns:
            score: 1 - Accuracy score on the test set
        """

        # Create model using the chosen hyps
        x_dict = x.to_dict()

        # results_path = self.results_path()
        # if os.path.exists(results_path):
        #     evaluations = safe_load_w_pickle(results_path)
        #     if str(x_dict) in evaluations:
        #         return evaluations[str(x_dict)]

        model = self.create_model(x_dict)

        # Train model
        model.fit(self.train_x, self.train_y)

        # Test model performance
        y_pred = model.predict(self.test_x)

        # 1-acc for minimization
        if self.task_type == 'clf':
            score = 1 - metrics.accuracy_score(self.test_y, y_pred)
        elif self.task_type == 'reg':
            score = metrics.mean_squared_error(self.test_y, y_pred)
        else:
            raise NotImplementedError

        # if not os.path.exists(results_path):
        #     os.makedirs(os.path.dirname(results_path), exist_ok=1)
        #     save_w_pickle({}, results_path)

        # evaluations = load_w_pickle(results_path)
        # evaluations[str(x_dict)] = score
        # save_w_pickle(evaluations, results_path)

        return score

    def create_model(self, xgboost_kwargs: Dict[str, Any]):

        if self.task_type == 'clf':
            model = xgboost.XGBClassifier(**xgboost_kwargs)
        elif self.task_type == 'reg':
            model = xgboost.XGBRegressor(**xgboost_kwargs)
        else:
            raise ValueError(self.task_type)
        return model

    @staticmethod
    def get_data(dataset_id) -> Bunch:
        if dataset_id == 'boston':
            data = datasets.load_boston()
        elif dataset_id == 'mnist':
            data = datasets.load_digits()
        else:
            raise NotImplementedError("Invalid choice of task")
        return data

    @staticmethod
    def get_task_type(dataset_id) -> str:
        if dataset_id in ['boston']:
            task_type = 'reg'
        elif dataset_id in ['mnist']:
            task_type = 'clf'
        else:
            raise NotImplementedError("Invalid choice of task")

        return task_type

    @staticmethod
    def get_static_search_space_params(dataset_id: str) -> List[Dict[str, Any]]:
        task_type = XGBoostTask.get_task_type(dataset_id)

        if task_type == 'clf':
            objectives = ['multi:softmax', 'multi:softprob']
        elif task_type == 'reg':
            objectives = ['reg:linear', 'reg:logistic', 'reg:gamma',
                          'reg:tweedie']
        else:
            raise ValueError(task_type)

        params = [
            {'name': 'booster', 'type': 'nominal', 'categories': ['gbtree', 'dart']},  # linear booster ignored
            {'name': 'grow_policy', 'type': 'nominal', 'categories': ['depthwise', 'lossguide']},
            {'name': 'objective', 'type': 'nominal', 'categories': objectives},
            {'name': 'learning_rate', 'type': 'pow', 'lb': 1e-5, 'ub': 1},
            {'name': 'max_depth', 'type': 'int', 'lb': 1, 'ub': 10},
            {'name': 'min_split_loss', 'type': 'num', 'lb': 0, 'ub': 10},
            {'name': 'subsample', 'type': 'num', 'lb': 0.001, 'ub': 1},
            {'name': 'reg_lambda', 'type': 'num', 'lb': 0, 'ub': 5},
        ]

        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params(dataset_id=self.dataset_id)

    def results_path(self) -> str:
        return self.get_result_path(dataset_id=self.dataset_id)

    @staticmethod
    def get_result_path(dataset_id: str) -> str:
        results_dir = str(Path(os.path.realpath(__file__)).parent)
        return os.path.join(results_dir, f"xgboost-{dataset_id}", "evaluations.pkl")
