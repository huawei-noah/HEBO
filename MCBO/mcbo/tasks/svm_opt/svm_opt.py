import os
import pathlib
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from mcbo.tasks import TaskBase


class SVMOptTask(TaskBase):
    """ Tuning of sklearn SVM regression learner hyperparamters on the slice localisation dataset """

    @property
    def name(self) -> str:
        return 'SVM Opt'

    def __init__(self, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.x_trains = None
        self.x_tests = None
        self.y_trains = None
        self.y_tests = None

    @staticmethod
    def prepare_x_y() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        dataset_path = os.path.join(str(pathlib.Path(__file__).parent.parent.resolve()), "data",
                                    "slice_localization_data.csv")
        data = pd.read_csv(dataset_path, sep=",").to_numpy()

        x = data[:, :-1]
        y = data[:, -1]

        # remove constant features
        features_to_keep = (x.max(0) - x.min(0)) > 1e-6
        x = x[:, features_to_keep]

        mixed_inds = np.random.RandomState(0).permutation(len(x))

        x = x[mixed_inds[:10000]]
        y = y[mixed_inds[:10000]]

        # select most important features using XGBoost
        feature_select_regr = XGBRegressor(max_depth=8).fit(x, y)
        feature_select_inds = np.argsort(feature_select_regr.feature_importances_)[::-1][:50]  # Keep 50 features
        x = x[:, feature_select_inds]

        # create train and test splits
        x_trains, y_trains, x_tests, y_tests = [], [], [], []

        for seed in range(5):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,
                                                                                        test_size=.3,
                                                                                        random_state=seed)
            x_trains.append(x_train)
            x_tests.append(x_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

        return x_trains, x_tests, y_trains, y_tests

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:
        """ Transform entry x into a RNA sequence and evaluate it's fitness given as the Hamming distance
            between the folded RNA sequence and the target
        """

        if self.x_trains is None or self.y_trains is None:
            self.x_trains, self.x_tests, self.y_trains, self.y_tests = self.prepare_x_y()

        evaluations = []
        for i in range(len(x)):
            svm_hyp = x.iloc[i]

            scores = []

            for j in range(5):
                x_train, x_test, y_train, y_test = self.x_trains[j], self.x_tests[j], self.y_trains[j], self.y_tests[j]

                # standardize y_train
                y_train_mean, y_train_std = y_train.mean(), y_train.std()
                y_train = (y_train - y_train_mean) / y_train_std

                # select features
                features_filter = np.array([np.round(getattr(svm_hyp, f"feature_{j + 1}")) for j in range(50)]).astype(
                    int)
                if np.sum(features_filter) == 0:  # nothing selected
                    y_pred = y_train_mean * np.ones(len(x_test))

                else:
                    x_train = x_train[:, features_filter]
                    x_test = x_test[:, features_filter]
                    learner = SVR(epsilon=svm_hyp.epsilon, C=svm_hyp.C, gamma=svm_hyp.gamma / x_train.shape[-1])
                    regr = make_pipeline(MinMaxScaler(), learner)
                    regr.fit(x_train, y_train)
                    y_pred = regr.predict(x_test) * y_train_std + y_train_mean
                scores.append(mean_squared_error(y_test, y_pred))
            evaluations.append(np.mean(scores))

        return np.array(evaluations).reshape(-1, 1)

    @staticmethod
    def get_static_search_space_params()  -> List[Dict[str, Any]]:
        """ Return search space params associated to this task """
        params = []
        for i in range(50):
            params.append({'name': f'feature_{i + 1}', 'type': 'nominal', 'categories': [0, 1]})

        params.extend([
            {'name': 'epsilon', 'type': 'pow', 'lb': 1e-2, 'ub': 1},
            {'name': 'C', 'type': 'pow', 'lb': 1e-2, 'ub': 100},
            {'name': 'gamma', 'type': 'pow', 'lb': 1e-1, 'ub': 10},
        ])

        return params

    def get_search_space_params(self) -> List[Dict[str, Any]]:
        return self.get_static_search_space_params()
