#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

FILE_PATH = "./workspace/hyperopt/ps821/data/"    

sample_submission = pd.read_csv(FILE_PATH+'sample_submission.csv.zip')
train = pd.read_csv(FILE_PATH+'train.csv.zip')
test = pd.read_csv(FILE_PATH+'test.csv.zip')


sample_submission.head()


train.head()


test.head()


columns = test.columns[1:]
columns


X = train[columns].values
X_test = test[columns].values
target = train['loss'].values.reshape(-1,1)



train_oof = np.zeros((train.shape[0],))
test_preds = np.zeros((test.shape[0],))
train_oof.shape


test_preds.shape


n_splits = 5
n_seeds = 16
from sklearn.metrics import mean_squared_error

model = HistGradientBoostingRegressor(max_iter=8700, learning_rate=0.01, early_stopping=False, max_depth=22)

# for seed in range(n_seeds):
#     kf = KFold(n_splits=n_splits, random_state=2*seed**3+137, shuffle=True)

#     for jj, (train_index, val_index) in enumerate(kf.split(train)):
#         print("Fitting fold", jj+1)
#         train_features = X[train_index]
#         train_target = target[train_index]


#         val_features = X[val_index]
#         val_target = target[val_index]


#         model = HistGradientBoostingRegressor(max_iter=8700, learning_rate=0.01, early_stopping=False, max_depth=22)
#         model.fit(train_features, train_target)
#         val_pred = model.predict(val_features)
#         train_oof[val_index] += val_pred.flatten()/n_seeds
#         test_preds += model.predict(X_test).flatten()/(n_splits*n_seeds)
# mean_squared_error(target,train_oof, squared=False)


# np.save('train_oof', train_oof)
# np.save('test_preds', test_preds)


# sample_submission['loss'] = test_preds
# sample_submission.to_csv('submission.csv', index=False)
# sample_submission.head()




