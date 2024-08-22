#!/usr/bin/env python
# coding: utf-8



import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.base import copy
import pandas as pd
import numpy as np
import random
# import optuna

RANDOM_SEED = 42

# This action may be dangerous for the private score
MAKING_ENSEMBLE = True

FIND_BEST_PARAMS = False
APPLY_LOG_TRANSFORMATION = True
APPLY_FEATURE_ENGINEERING = True

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
FILE_PATH='./workspace/hyperopt/abalone/data/'
# FILE_PATH='../data/'
submmision_file='submission.csv'
train=pd.read_csv(FILE_PATH+'train.csv')
orginal =pd.read_csv(FILE_PATH+'abalone.csv')
test=pd.read_csv(FILE_PATH+'test.csv')


train = train.drop(['id'], axis = 1)
train.columns = orginal.columns
train = pd.concat([train, orginal], axis = 0, ignore_index=True)

y = train['Rings']
# Because RMSLE score, We make a conversion like below:
y_log = np.log(1+y)
# Add the end for getting the result we back to original like below:
# y = np.exp(y_log)-1


train = train.drop(['Rings'], axis = 1)
train.head()

test_id = test['id']
test = test.drop('id', axis = 1)
test.columns = train.columns
test.head()

encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')

train = pd.concat([
                    train.iloc[:,1:], 
                    pd.DataFrame(encoder.fit_transform(train[['Sex']]).astype('int'), 
                                 columns = encoder.categories_[0])
                    ], 
                    axis = 1
                )

test  = pd.concat([
                    test.iloc[:,1:], 
                    pd.DataFrame(encoder.transform(test[['Sex']]).astype('int'), 
                                 columns = encoder.categories_[0])
                    ], 
                    axis = 1
                )

def log_transformation(data, columns):
    for column in columns:
        positive_values = data[column] - data[column].min() + 1
        data[f'{column}_log'] = np.log(positive_values)
    return data


if APPLY_LOG_TRANSFORMATION:
    train = log_transformation(train, ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight', 'Shell weight'])
    test  = log_transformation(test, ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight','Viscera weight', 'Shell weight'])



# def objective(trial):

#     params = {
#         "verbose": False,
#         "iterations": 1000,
#         "loss_function":'RMSE',
#         "random_state": RANDOM_SEED,
#         "depth": trial.suggest_int("depth", 3, 15),
#         "subsample": trial.suggest_float("subsample", 0.01, 1.0),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
#         "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
#     }

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
#     scores = []
#     for _, (train_index, valid_index) in enumerate(cv.split(train, y)):
#         X_train, y_train = train.iloc[train_index], y_log.iloc[train_index]
#         X_valid, y_valid = train.iloc[valid_index], y_log.iloc[valid_index]
#         model = CatBoostRegressor(**params)

#         model.fit(X_train, y_train, 
#                   eval_set=(X_valid, y_valid),
#                   early_stopping_rounds=100)
        
#         y_pred = model.predict(X_valid)
#         scores.append(root_mean_squared_error(y_valid, y_pred))
#     return np.mean(scores)


# study = optuna.create_study(direction='minimize', study_name="optuna_catboost")
# if FIND_BEST_PARAMS:
#     study.optimize(objective, n_trials=50)
#     print(f"Best trial average RMSE: {study.best_value:.4f}")
#     for key, value in study.best_params.items():
#         print(f"{key}: {value}")



# def objective(trial):

#     params = {
#         'n_jobs':-1,
#         "metric":'rmse',  
#         "verbosity": -1,
#         "bagging_freq": 1,
#         "boosting_type": "gbdt",    
#         "objective":'regression', 
#         'random_state':RANDOM_SEED,
#         'max_depth': trial.suggest_int('max_depth', 3, 15),                        
#         "subsample": trial.suggest_float("subsample", 0.05, 1.0),
#         "n_estimators": trial.suggest_int('n_estimators', 400, 1000),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),               
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.01),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
#     }

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
#     scores = []
#     for _, (train_index, valid_index) in enumerate(cv.split(train, y)):
#         X_train, y_train = train.iloc[train_index], y_log.iloc[train_index]
#         X_valid, y_valid = train.iloc[valid_index], y_log.iloc[valid_index]
#         model = LGBMRegressor(**params)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_valid)     
#         scores.append(root_mean_squared_error(y_valid, y_pred))
#     return np.mean(scores)


# study = optuna.create_study(direction='minimize', study_name="optuna_lgbm")
# if FIND_BEST_PARAMS:
#     study.optimize(objective, n_trials=50)
#     print(f"Best trial average RMSE: {study.best_value:.4f}")
#     for key, value in study.best_params.items():
#         print(f"{key}: {value}")


# def objective(trial):

#     params = {
#         'eval_metric': 'rmse',
#         'random_state': RANDOM_SEED,
#         'objective': 'reg:squarederror',
#         'gamma': trial.suggest_float("gamma", 1e-2, 1.0),
#         'max_depth': trial.suggest_int('max_depth',2, 20),
#         'subsample': trial.suggest_float("subsample", 0.05, 1.0),
#         'n_estimators': trial.suggest_int('n_estimators',100, 1000),
#         'min_child_weight': trial.suggest_int('min_child_weight',2, 20),
#         'colsample_bytree': trial.suggest_float("colsample_bytree", 0.05, 1.0),
#         'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
#     }

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
#     scores = []
#     for _, (train_index, valid_index) in enumerate(cv.split(train, y)):
#         X_train, y_train = train.iloc[train_index], y_log.iloc[train_index]
#         X_valid, y_valid = train.iloc[valid_index], y_log.iloc[valid_index]
#         model = XGBRegressor(**params)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_valid)     
#         scores.append(root_mean_squared_error(y_valid, y_pred))
#     return np.mean(scores)


# study = optuna.create_study(direction='minimize', study_name="optuna_xgboost")
# if FIND_BEST_PARAMS:
#     study.optimize(objective, n_trials=50)
#     print(f"Best trial average RMSE: {study.best_value:.4f}")
#     for key, value in study.best_params.items():
#         print(f"{key}: {value}")

xgboost_params = {
    'max_depth': 10, 
    'verbosity': 0,
    'random_state':RANDOM_SEED,
    'device': 'cuda',
    'booster': 'gbtree',
    'n_estimators': 1137, 
    'tree_method': 'hist',
    'min_child_weight': 7, 
    'grow_policy': 'lossguide', 
    'gamma': 0.03816426816838989, 
    'subsample': 0.486382907668344, 
    'objective': 'reg:squarederror',
    'reg_lambda': 1.7487237399420372, 
    'reg_alpha': 0.013043045359306716,
    'learning_rate': 0.011733966748427322, 
    'colsample_bytree': 0.5748511749872887, 
}

lgbm_params = {
     'metric':'rmse', 
     'device':'cpu', 
     'verbosity': -1,
     'max_depth': 15,
     'random_state':RANDOM_SEED,
     'num_leaves': 138, 
     'n_estimators': 913, 
     'boosting_type': 'gbdt', 
     'min_child_samples': 34, 
     'objective':'regression', 
     'subsample_for_bin': 185680, 
     'subsample': 0.799314727120346, 
     'reg_alpha': 5.916235901972299e-09, 
     'reg_lambda': 6.943912907338958e-08, 
     'learning_rate': 0.01851440025520457, 
     'colsample_bytree': 0.4339090795122026, 
}

catboost_params = {
    'depth': 15, 
    'max_bin': 464, 
    'verbose': False,
    'random_state':RANDOM_SEED,
    'task_type': 'CPU', 
    'eval_metric': 'RMSE', 
    'min_data_in_leaf': 78, 
    'loss_function': 'RMSE', 
    'grow_policy': 'Lossguide', 
    'bootstrap_type': 'Bernoulli', 
    'subsample': 0.83862137638162, 
    'l2_leaf_reg': 8.365422739510098, 
    'random_strength': 3.296124856352495, 
    'learning_rate': 0.09992185242598203, 
}

cv_estimators = [
    ('lgbm', LGBMRegressor(**lgbm_params)),
    ('xgboost', XGBRegressor(**xgboost_params)),
    ('catboost', CatBoostRegressor(**catboost_params))
]


# def objective(trial):
    
#     params = {
#         'lgbm_weight': trial.suggest_float('lgbm_weight', 0.0, 5.0),
#         'xgboost_weight': trial.suggest_float('xgboost_weight', 0.0, 5.0),
#         'catboost_weight': trial.suggest_float('catboost_weight', 0.0, 5.0),
#     }


#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
#     scores = []
#     for _, (train_index, valid_index) in enumerate(cv.split(train, y)):
#         X_train, y_train = train.iloc[train_index], y_log.iloc[train_index]
#         X_valid, y_valid = train.iloc[valid_index], y_log.iloc[valid_index]
#         voting_regressor = VotingRegressor(
#             estimators=cv_estimators,
#             weights=[params['lgbm_weight'], params['xgboost_weight'], params['catboost_weight']]
#         )
#         voting_regressor.fit(X_train, y_train)
#         y_pred = voting_regressor.predict(X_valid)  
#         scores.append(root_mean_squared_error(y_valid, y_pred))
#     return np.mean(scores)


# study = optuna.create_study(direction='minimize', study_name="voting_regressor_optuna")
# if FIND_BEST_PARAMS:
#     study.optimize(objective, n_trials=100)
#     print(f"Best trial average RMSE: {study.best_value:.4f}")
#     for key, value in study.best_params.items():
#         print(f"{key}: {value}")



train2 = train.copy()
test2  = test.copy()

# I find these drop cols with feature selection base genetic algorithm
lst_drop_cols = [
    ['Shucked weight', 'Shell weight', 'Length_log', 'Diameter_log', 'Height_log', 'Viscera weight_log'],
                 ['Shell weight', 'I', 'Length_log', 'Height_log', 'Viscera weight_log']]

lst_y_pred_test = []
# for i in range(len(lst_drop_cols)):
#     if APPLY_FEATURE_ENGINEERING:
#         train2 = train.drop(lst_drop_cols[i], axis=1)
#         test2  = test.drop(lst_drop_cols[i], axis=1)

weight_best_params = {
    'lgbm_weight': 4.104966149239676, 
    'xgboost_weight': 0.48550637896530635, 
    'catboost_weight': 4.189724537494019,
}


voting_regressor = VotingRegressor(
    estimators=cv_estimators,
    weights=[ 
        weight_best_params['lgbm_weight'], 
                weight_best_params['xgboost_weight'], 
                weight_best_params['catboost_weight']
    ]
)

# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
# scores = []
# y_pred_test = []
# for fold_i, (train_index, valid_index) in enumerate(cv.split(train, y)):
#     X_train, y_train = train2.iloc[train_index], y_log.iloc[train_index]
#     X_valid, y_valid = train2.iloc[valid_index], y_log.iloc[valid_index]
#     voting_regressor.fit(X_train, y_train)
#     y_pred = voting_regressor.predict(X_valid)  
#     score = root_mean_squared_error(y_valid, y_pred)
#     scores.append(score)
#     y_pred_test.append(voting_regressor.predict(test2))
#     print(f"FOLD {fold_i} Done. RMSE : {score}")
# print(f"All FOLD. Mean RMSE : {np.mean(scores)}")


# lst_y_pred_test.append(np.mean(y_pred_test, axis=0))

# predictions = np.mean(lst_y_pred_test, axis=0)
# sub  = pd.DataFrame(columns = ['id', 'Rings'])
# sub['id'] = test_id
# sub['Rings'] = np.exp(predictions)-1
# sub.to_csv(FILE_PATH+submmision_file, index = False)

# score = np.mean(scores)
