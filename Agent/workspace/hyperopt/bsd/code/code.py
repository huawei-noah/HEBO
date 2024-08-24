#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

from audioop import rms
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won"t be saved outside of the current session


# Version 1.0

FILE_PATH = "../data/"
# FILE_PATH= "./workspace/hyperopt/bsd/data/"

submission_path = "ori_submission.csv"
RANDOM_SEED = 73

import pandas as pd

train = pd.read_csv(FILE_PATH + "train.csv")
test = pd.read_csv(FILE_PATH + "test.csv")

# train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=["datetime"])
# test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=["datetime"])
train.shape, test.shape

train.info()

train.describe()

train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])
train.info(), test.info()

train.corr()

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
# train["day"] = train["datetime"].dt.day
train["dayofweek"] = train["datetime"].dt.dayofweek
# train["weekday"] = train["datetime"].dt.weekday
train["hour"] = train["datetime"].dt.hour
# dt.quarter
# dt.minute
# dt.seconds

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["dayofweek"] = test["datetime"].dt.dayofweek
test["hour"] = test["datetime"].dt.hour

X_train = train.drop(["datetime", "casual", "registered", "count", ], axis=1)
y_train = train["count"]
test_id = test["datetime"]
test = test.drop(["datetime"], axis=1)

y_train.min()

cols = X_train.select_dtypes(exclude="object").columns
cols

# import sklearn.preprocessing
# sklearn.preprocessing.__all__


from sklearn.preprocessing import MinMaxScaler
# for col in cols:
#     mms = MinMaxScaler()
#     X_train[col] = mms.fit_transform(X_train[[col]])
#     test[col] = mms.transform(test[[col]])


from sklearn.preprocessing import StandardScaler
# for col in cols:
#     ss = StandardScaler()
#     X_train[col] = ss.fit_transform(X_train[[col]])
#     test[col] = ss.transform(test[[col]])


# X_train


from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# X_train[cols] = ss.fit_transform(X_train[cols])
# X_train[cols] = ss.transform(X_train[cols])


# X_train


from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=RANDOM_SEED)
# model = rfr.fit(X_tr, y_tr)
# pred = model.predict(X_val)


# # import xgboost
# # dir(xgboost)


from xgboost import XGBRegressor
# # xgbr = XGBRegressor(random_state=2024)
# # model = xgbr.fit(X_tr, y_tr)
# # pred = model.predict(X_val)


# # import lightgbm
# # lightgbm.__all__


from lightgbm import LGBMRegressor
# # lgbmr = LGBMRegressor(random_state=2024)
# # model = lgbmr.fit(X_tr, y_tr)
# # pred = model.predict(X_val)


# pred.min()


# # import sklearn.metrics
# # sklearn.metrics.__all__


# # import sklearn.metrics
# # help (sklearn.metrics.mean_squared_log_error)


# rmsle = mean_squared_log_error(y_val, pred, squared=False)


# y_pred = model.predict(test)
# y_pred.min()


# submit = pd.DataFrame({"datetime" : test_id, "count" : y_pred})
# submit.to_csv(submission_path, index=False)


# score=rmsle
