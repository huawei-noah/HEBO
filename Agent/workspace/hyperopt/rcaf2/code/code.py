#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# FILE_PATH="../data/"
FILE_PATH = "./workspace/hyperopt/rcaf2/data/"
# TARGET = "NObeyesdad"
submission_path = "ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73


# Any results you write to the current directory are saved as output.


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


train = import_data(FILE_PATH + "train.csv")
test = import_data(FILE_PATH + "test.csv")
# train.head()


test_id = test["id"]
test.drop(["id"], axis=1, inplace=True)

import lightgbm as lgb

dic = {"A": 0, "B": 1, "C": 2, "D": 3}
dic1 = {"CA": 0, "DA": 1, "SS": 3, "LOFT": 4}
train["event"] = train["event"].apply(lambda x: dic[x])
train["event"] = train["event"].astype("int8")
train["experiment"] = train["experiment"].apply(lambda x: dic1[x])
test["experiment"] = test["experiment"].apply(lambda x: dic1[x])

train["experiment"] = train["experiment"].astype("int8")
test["experiment"] = test["experiment"].astype("int8")

# train.info()
y = train["event"]
train.drop(["event"], axis=1, inplace=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.4, random_state=42)

# ***Imbalance learning***

from imblearn.over_sampling import SMOTE

X_train, y_train = SMOTE().fit_resample(X_train, y_train.ravel())

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb

clf1 = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05)
clf2 = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1)
clf3 = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.2)

# clf1.fit(X_train, y_train)
# pred1 = clf1.predict(X_test)
# print("lgbm1: ", accuracy_score(pred1, y_test))

# pred = clf1.predict_proba(test)
# sub = pd.DataFrame(pred, columns=["A", "B", "C", "D"])
# sub["id"] = test_id
# cols = sub.columns.tolist()
# cols = cols[-1:] + cols[:-1]
# sub = sub[cols]
# sub.to_csv("sub_lgb1.csv", index=False)

# clf2.fit(X_train, y_train)
# pred2 = clf2.predict(X_test)
# print("lgbm2: ", accuracy_score(pred2, y_test))

# clf3.fit(X_train, y_train)
# pred3 = clf3.predict(X_test)
# print("lgbm3: ", accuracy_score(pred3, y_test))
voting_classifier_fix_params={
    "estimators": [("lr", clf1), ("rf", clf2), ("gnb", clf3)],
    "voting": "soft",
    "weights": [2, 1, 1],
    "flatten_transform": True
}
eclf3 = VotingClassifier(**voting_classifier_fix_params)
