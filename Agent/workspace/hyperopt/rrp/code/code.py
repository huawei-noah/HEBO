#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# FILE_PATH="../data/"
FILE_PATH = "./workspace/hyperopt/rrp/data/"
TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
n_splits = 9
RANDOM_SEED = 73

import pandas as pd

df_train = pd.read_csv(FILE_PATH + "train.csv.zip")

df_test = pd.read_csv(FILE_PATH + "test.csv.zip")
df_test.head()

df_test.shape

# # **訓練データの前処理**

import datetime
from sklearn.preprocessing import LabelEncoder

# Open Dateを年,月,日に分解
df_train["Open Date"] = pd.to_datetime(df_train["Open Date"])
df_train["Year"] = df_train["Open Date"].apply(lambda x: x.year)
df_train["Month"] = df_train["Open Date"].apply(lambda x: x.month)
df_train["Day"] = df_train["Open Date"].apply(lambda x: x.day)

# Restaurant Revenue PredictionがKaggleに立ち上げられた日
df_train["launch"] = "2015-03-24"
df_train["launch"] = pd.to_datetime(df_train["launch"])

# 店の営業日数
df_train["workday"] = (df_train["launch"] - df_train["Open Date"]).apply(lambda x: x.days)
df_train.drop(["Open Date", "launch"], axis=1, inplace=True)

# 文字列を数値に変換
label = LabelEncoder()
df_train["City"] = label.fit_transform(df_train["City"])
df_train["City Group"] = df_train["City Group"].map({"Other": 0, "Big Cities": 1})
df_train["Type"] = df_train["Type"].map({"FC": 0, "IL": 1, "DT": 2, "MB": 3})

df_train

import copy

df_train_corr = copy.copy(df_train)
df_train_corr.drop(["Id", "City", "City Group", "Type"], axis=1, inplace=True)
df_train_corr.drop(["Year", "Month", "Day", "workday"], axis=1, inplace=True)

df_train_revenue_corr = list(df_train.corr()["revenue"])

# df_train_revenue_corr.drop(["revenue"], axis=1, inplace=True)

df_train_revenue_corr

df_train_revenue_corr[0:4] = []
df_train_revenue_corr[37:42] = []
df_train_revenue_corr

np.abs(df_train_revenue_corr)

import matplotlib.pyplot as plt

labels = df_train_corr.columns
corr = np.abs(df_train_revenue_corr)

# # **訓練データの中で数値の大きさが本質でない変数を2進数にダミー化**

training = pd.get_dummies(df_train, columns=["City", "City Group", "Type"], drop_first=True)
training

# **店舗が1つしかない都市を削除**

training.drop(["City_1", "City_2", "City_6", "City_7"], axis=1, inplace=True)
training.drop(["City_9", "City_11", "City_12", "City_14"], axis=1, inplace=True)
training.drop(["City_15", "City_16", "City_17", "City_19"], axis=1, inplace=True)
training.drop(["City_21", "City_22", "City_24", "City_28"], axis=1, inplace=True)
training.drop(["City_30", "City_33"], axis=1, inplace=True)

# # **テストデータの前処理**

# Open Dateを年,月,日に分解
df_test["Open Date"] = pd.to_datetime(df_test["Open Date"])
df_test["Year"] = df_test["Open Date"].apply(lambda x: x.year)
df_test["Month"] = df_test["Open Date"].apply(lambda x: x.month)
df_test["Day"] = df_test["Open Date"].apply(lambda x: x.day)

# Restaurant Revenue PredictionがKaggleに立ち上げられた日
df_test["launch"] = "2015-03-24"
df_test["launch"] = pd.to_datetime(df_test["launch"])

# 店の営業日数
df_test["workday"] = (df_test["launch"] - df_test["Open Date"]).apply(lambda x: x.days)
df_test.drop(["Open Date", "launch"], axis=1, inplace=True)

# 文字列を数値に変換
label = LabelEncoder()
df_test["City"] = label.fit_transform(df_test["City"])
df_test["City Group"] = df_test["City Group"].map({"Other": 0, "Big Cities": 1})
df_test["Type"] = df_test["Type"].map({"FC": 0, "IL": 1, "DT": 2, "MB": 3})

df_test

training.columns

# # **テストデータの中で数値の大きさが本質でない変数を2進数にダミー化**

test = pd.get_dummies(df_test, columns=["City", "City Group", "Type"], drop_first=True)

# # **ランダムフォレストで学習**

from sklearn.ensemble import RandomForestRegressor

# 学習に使う特徴量を取得
cols = ["P1", "P2", "P6", "P8", "P10", "P11", "P12", "P13", "P17", "P21", "P22", "P28", "P29", "P30", "P32", "P34",
        "workday", "City_3", "City_4", "City_5", "City_8", "City_10", "City_13", "City_18", "City_20", "City_23",
        "City_25", "City_26", "City_27", "City_29", "City_31", "City_32", "City Group_1", "Type_1", "Type_2"]
X_train = training[cols]
y_train = training["revenue"]

X_test = test[cols]
# y_test = df_test["revenue"]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
# print(y_test.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)
# RandomForestで学習させる
rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X_train, y_train)


# # **学習したモデルをテストデータに適用**

# from sklearn import metrics
# prediction = rf.predict(X_test)
# #print(metrics.classification_report(y_train, prediction))
# prediction


# # 予測結果をデータフレーム型へ変換
# df = pd.DataFrame(prediction, columns=["Prediction"])
# # id列を作成
# df.insert(0, "Id", range(1, len(prediction) + 1))
# df # 確認用


# # # **提出ファイルに予測データを保存**

# # ひな形ファイルを読み込む
# sub = pd.read_csv(FILE_PATH+"sampleSubmission.csv")
# print(f"~~ひな形の内容~~\n{sub}")
# # subのPrediction列をdfのPrediction列で上書きする
# sub["Prediction"] = df["Prediction"]
# # 結果を表示
# print(f"~~予測結果の内容~~\n{sub}")
# sub.to_csv(submission_path, index=False) # 変換したファイルを保存

# score=1.0 - rf.score(X_val, y_val)
# print(score)
