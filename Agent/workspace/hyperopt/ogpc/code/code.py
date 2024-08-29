#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
# import optuna.integration.lightgbm as lgb
from sklearn.metrics import accuracy_score
import seaborn as sns

FILE_PATH = "./workspace/hyperopt/ogpc/data/"

train = pd.read_csv(FILE_PATH+'train.csv.zip')
test = pd.read_csv(FILE_PATH+'test.csv.zip')
sample_submit = pd.read_csv(FILE_PATH+'sampleSubmission.csv.zip')


# replace 'Class_1' ~ 'Class_9' to 0~8
train['target'] = train['target'].str.replace('Class_', '')
train['target'] = train['target'].astype(int) - 1


# # Feature Engineering

data = pd.concat([train, test])
cols = [c for c in data.columns if c not in ['id', 'target']]

for col in cols:
    dictionary=data[col].value_counts().to_dict()
    data['count_'+col]=data[col].map(dictionary)
from sklearn import preprocessing

data['max_val'] = data[cols].max(axis=1)
data['sum_val'] = data[cols].sum(axis=1)
data['non_zero'] = (data[cols] > 0).sum(axis=1)
data['count_one'] = (data[cols] == 1).sum(axis=1)
data['count_two'] = (data[cols] == 2).sum(axis=1)
data['count_three'] = (data[cols] == 3).sum(axis=1)

train = data[~data['target'].isnull()].reset_index(drop=True)
test = data[data['target'].isnull()].reset_index(drop=True)


# train setting
NFOLDS = 5
RANDOM_STATE = 871972

excluded_column = ['target', 'id']
cols = [c for c in train.columns if c not in excluded_column]

# parameter calculated by LGBtuner
# params = {
#     'metric':'multi_logloss','objective': 'multiclass',   'num_class': 9,   'verbosity': 1,
#     'feature_fraction': 0.4, 'num_leaves': 139, 'bagging_fraction': 0.8254401463359962, 'bagging_freq': 3,
#     'lambda_l1': 0.02563829140437355, 'lambda_l2': 9.594334397031103, 'min_child_samples': 100
# }


y_pred_test = np.zeros((len(test), 9))
oof = np.zeros((len(train), 9))
score = 0
feature_importance_df = pd.DataFrame()

lgb_model=LGBMClassifier()
# for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y = train['target'])):
#     print('Fold', fold_n)
#     X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
#     y_train, y_valid = X_train['target'].astype(int), X_valid['target'].astype(int)
    
#     train_data = lgb.Dataset(X_train[cols], label=y_train)
#     valid_data = lgb.Dataset(X_valid[cols], label=y_valid)

#     lgb_model = lgb.train(params,train_data,num_boost_round=30000,
#                     valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 300)
    
#     y_pred_valid = lgb_model.predict(X_valid[cols], num_iteration=lgb_model.best_iteration)
#     oof[valid_index] = y_pred_valid
#     score += log_loss(y_valid, y_pred_valid)
    
#     y_pred_test += lgb_model.predict(test[cols], num_iteration=lgb_model.best_iteration)/NFOLDS
    
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["feature"] = cols
#     fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
#     fold_importance_df["fold"] = fold_n + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
# print('valid logloss average:', score/NFOLDS, log_loss(train['target'], oof))


# feature_importance_df[["feature", "importance"]].groupby("feature", as_index=False).mean().sort_values(by="importance", ascending=False).head(20)


# submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_test)], axis = 1)
# submit.columns = sample_submit.columns
# submit.to_csv('submit.csv', index=False)


# column_name = ['lgb_' + str(i) for i in range(9)]
# pd.DataFrame(oof, columns = column_name).to_csv('oof_lgb.csv', index=False)
# pd.DataFrame(y_pred_test, columns = column_name).to_csv('submit_lgb.csv', index=False)




