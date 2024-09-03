#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# The purpose of this notebook is to predict the target by an Ensemble model composed of tree individual models
# 
# - lightgbm
# - xgboost
# - catboost
# 
# Feature Engineering followed basic practices that proved to work for GBM-style models for this competition
# 
# - label encoding the cat variables
# - standard scaling to numeric variables
# 
# Params for *xgboost* and *catboost* have been discovered via hyperparam search, using *hyperopt*. Params for *lightgbm* have been reused from https://www.kaggle.com/hiro5299834/tps-feb-2021-with-single-lgbm-tuned (they appeared to work better vs. the set of parameters I discovered in *hyperopt*-based search).
# 
# Weight of lightgbm prediction was set to be a little higher then catboost and xgboost.
# 
# The well-thought software design of the Ensembling class was inspired by https://www.kaggle.com/kenkpixdev/ensemble-lgb-xgb-with-hyperopt

import pandas as pd
import numpy as np
import time
import datetime as dt
from typing import Tuple, List, Dict

# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
# from hyperopt.pyll.base import scope
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder


# main flow
start_time = dt.datetime.now()
print("Started at ", start_time)


# read data
in_kaggle = True

FILE_PATH = "./workspace/hyperopt/tspf2/data/"

def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str, str]:
    train_path = ''
    test_path = ''
    sample_submission_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/tabular-playground-series-feb-2021/train.csv'
        test_path = '../input/tabular-playground-series-feb-2021/test.csv'
        sample_submission_path = '../input/tabular-playground-series-feb-2021/sample_submission.csv'
    else:
        # running locally
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
        sample_submission_path = 'data/sample_submission.csv'

    return train_path, test_path, sample_submission_path


# get_ipython().run_cell_magic('time', '', '# get the training set and labels\ntrain_set_path, test_set_path, sample_subm_path = get_data_file_path(in_kaggle)\n\ntrain = pd.read_csv(train_set_path)\ntest = pd.read_csv(test_set_path)\ntarget = train.target\n\nsubm = pd.read_csv(sample_subm_path)\n')


train = pd.read_csv(FILE_PATH + 'train.csv.zip')
test = pd.read_csv(FILE_PATH +'test.csv.zip')
target = train.target

subm = pd.read_csv(FILE_PATH + 'sample_submission.csv.zip')


def preprocess(df, encoder=None,scaler=None, cols_to_drop=None,cols_to_encode=None, cols_to_scale=None):
    """
    Preprocess input data
    :param df: DataFrame with data
    :param encoder: encoder object with fit_transform method
    :param scaler: scaler object with fit_transform method
    :param cols_to_drop: columns to be removed
    :param cols_to_encode: columns to be encoded
    :param cols_to_scale: columns to be scaled
    :return: DataFrame
    """

    if encoder: 
        for col in cols_to_encode:
            df[col] = encoder.fit_transform(df[col])

    if scaler:
        for col in cols_to_scale:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)

    return df


cat_cols = ['cat' + str(i) for i in range(10)]
cont_cols = ['cont' + str(i) for i in range(14)]

train = preprocess(train, encoder=LabelEncoder(), scaler=StandardScaler(),
                  cols_to_drop=['id', 'target'], cols_to_encode=cat_cols,
                  cols_to_scale=cont_cols)

# encoder=LabelEncoder()
test = preprocess(test, encoder=LabelEncoder(), scaler=StandardScaler(),
                 cols_to_drop=['id'], cols_to_encode=cat_cols,
                 cols_to_scale=cont_cols)

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
N_FOLDS = 10
N_ESTIMATORS = 30000
SEED = 2021
BAGGING_SEED = 48



class EnsembleModel:
    def __init__(self, params):
        """
        LGB + XGB + CatBoost model
        """
        self.lgb_params = params['lgb']
        self.xgb_params = params['xgb']
        self.cat_params = params['cat']

        self.lgb_model = LGBMRegressor(**self.lgb_params,
                                       **{
                                            'n_jobs': -1,
                                            'cat_feature': [x for x in range(len(cat_cols))],
                                            'bagging_seed': SEED,
                                            'feature_fraction_seed': SEED,
                                            'random_state': SEED,
                                            'metric': 'rmse',   
                                       }
                                       
                                       )
        self.xgb_model = XGBRegressor(**self.xgb_params,
                                      **{   'random_state': SEED,
                                            'objective': 'reg:squarederror',
                                            'tree_method': 'gpu_hist',
                                            'eval_metric': 'rmse',
                                            'n_jobs': -1
                                            }
                                      )



        self.cat_model = CatBoostRegressor(**self.cat_params,
                                           **{
                                            'random_state': SEED,
                                            'eval_metric': 'RMSE',
                                            'leaf_estimation_backtracking': 'AnyImprovement',

                                           }
                                           
                                           )

    def fit(self, x, y, *args, **kwargs):
        return (self.lgb_model.fit(x, y, *args, **kwargs),
                self.xgb_model.fit(x, y, *args, **kwargs),
               self.cat_model.fit(x, y, *args, **kwargs))

    def predict(self, x, weights=[1.0, 1.0, 1.0]):
        """
        Generate model predictions
        :param x: data
        :param weights: weights on model prediction, first one is the weight on lgb model
        :return: array with predictions
        """
        return (weights[0] * self.lgb_model.predict(x) +
                weights[1] * self.xgb_model.predict(x) +
                weights[2] * self.cat_model.predict(x)) / 3


since = time.time()
columns = train.columns



# ------------------------------------------------------------------------------
# LightGBM: training and inference
# ------------------------------------------------------------------------------
#

ensemble_params = {
    "lgb" : {
          'n_estimators': N_ESTIMATORS,
          'learning_rate': 0.003899156646724397,
          'max_depth': 99,
          'num_leaves': 63,
          'reg_alpha': 9.562925363678952,
          'reg_lambda': 9.355810045480153,
          'colsample_bytree': 0.2256038826485174,
          'min_child_samples': 290,
          'subsample_freq': 1,
          'subsample': 0.8805303688019942,
          'max_bin': 882,
          'min_data_per_group': 127,
          'cat_smooth': 96,
          'cat_l2': 19
          },
    'xgb': {
        'max_depth': 13,
        'learning_rate': 0.020206705089028228,
        'gamma': 3.5746731812451156,
        'min_child_weight': 564,
        'n_estimators': 8000,
        'colsample_bytree': 0.5015940592112956,
        'subsample': 0.6839489639112909,
        'reg_lambda': 18.085502002853246,
        'reg_alpha': 0.17532087359570606,
    },
    'cat': {
        'depth': 3.0,
        'fold_len_multiplier': 1.1425259013471902,
        'l2_leaf_reg': 7.567589781752637,
        'learning_rate': 0.25121635918496565,
        'max_bin': 107.0,
        'min_data_in_leaf': 220.0,
        'random_strength': 3.2658690042589726,
        'n_estimators': 8000,

    }
}
    
preds = np.zeros(test.shape[0])
kf = KFold(n_splits=N_FOLDS, random_state=22, shuffle=True)
rmse = []
n = 0
model = EnsembleModel(ensemble_params)
from sklearn.metrics import mean_squared_error
# for trn_idx, test_idx in kf.split(train[columns], target):

#     X_tr, X_val=train[columns].iloc[trn_idx], train[columns].iloc[test_idx]
#     y_tr, y_val=target.iloc[trn_idx], target.iloc[test_idx]

#     model = EnsembleModel(ensemble_params)

#     model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

#     preds += model.predict(test[columns], weights=[1.1, 1.0, 0.9]) / kf.n_splits
#     rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    
#     print(f"Fold {n+1}, RMSE: {rmse[n]}")
#     n += 1


# print("Mean RMSE: ", np.mean(rmse))
# end_time = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(
#         end_time // 60, end_time % 60))


# # submit prediction
# subm['target'] = preds
# subm.to_csv("ensemble_model_lgb_xgb_cat_other_lgb_params.csv", index=False)


# print('We are done. That is all, folks!')
# finish_time = dt.datetime.now()
# print("Finished at ", finish_time)
# elapsed = finish_time - start_time
# print("Elapsed time: ", elapsed)

