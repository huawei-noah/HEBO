#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

FILE_PATH = "./workspace/hyperopt/bank-churn/data/"

TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
n_splits = 9
RANDOM_SEED = 73

train = pd.read_csv(FILE_PATH + "train.csv")
original_train = pd.read_csv(FILE_PATH + "Churn_Modelling.csv")
test = pd.read_csv(FILE_PATH + "test.csv")
sample = pd.read_csv(FILE_PATH + "sample_submission.csv")

train.describe().T

original_train.describe().T

for cols in train.columns:
    print(f"The number of unique values in {cols} are {train[cols].nunique()}")

for cols in original_train.columns:
    print(f"The number of unique values in {cols} are {original_train[cols].nunique()}")

train.info()

original_train.drop("RowNumber", axis=1, inplace=True)
train = pd.concat([train, original_train], ignore_index=True)
train.drop_duplicates()

train.dropna(inplace=True)

train.isna().sum()

# # BASIC EDA

categorical_columns = train.select_dtypes(include="object").columns.tolist()
numerical_columns = train.select_dtypes(exclude="object").columns.tolist()

categorical_columns

# Plotting Distribution on one graph
sns.kdeplot(train[numerical_columns])


def plot_kde_for_all_columns(df):
    sns.set(style="whitegrid")
    columns = df.columns
    num_cols = 2
    num_rows = math.ceil(len(columns) / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2 * num_rows))
    axes = axes.flatten()
    for i, column in enumerate(columns):
        ax = axes[i]
        sns.kdeplot(df[column], ax=ax, fill=True)
        ax.set_title(f"KDE Plot for {column}")
        ax.set_xlabel(column)
    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()


plot_kde_for_all_columns(train[numerical_columns])

train

# # Approach 1 
# 
# ## Drop the id, CustomerID, Surname columns and then LabelEncode the other categorical columns

# train.drop(["id", "CustomerId", "Surname"], axis=1, inplace = True)
# test.drop(["id", "CustomerId", "Surname"], axis=1, inplace = True)


train.drop(["id"], axis=1, inplace=True)
test.drop(["id"], axis=1, inplace=True)

categorical_columns = train.select_dtypes(include="object").columns.tolist()

label_encoder = LabelEncoder()
for cols in categorical_columns:
    train[cols] = label_encoder.fit_transform(train[cols])
    test[cols] = label_encoder.fit_transform(test[cols])

# # Modelling

y = train.pop("Exited")
X = train

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# Below are the parameters for xgboost.

xgb_params = {"booster": "gbtree",
              "lambda": 0.8611971458776956,
              "alpha": 3.3684132992886347e-07,
              "max_depth": 3,
              "eta": 0.17374299923922656,
              "gamma": 1.2505690952357777e-06,
              "colsample_bytree": 0.8361517621930924,
              "min_child_weight": 2.650197692280842,
              "subsample": 0.645735940099335,
              "n_estimators": 137}

xgb_model = xgb.XGBClassifier(**xgb_params)
# xgb_model.fit(X,y)


lgb_params = {"n_estimators": 5000,
              "max_depth": 50,
              "learning_rate": 0.03,
              "min_child_weight": 0.81,
              "min_child_samples": 190,
              "subsample": 0.88,
              "subsample_freq": 2,
              "random_state": 42,
              "colsample_bytree": 0.62,
              "num_leaves": 15}

lgbm_model = lgbm.LGBMClassifier(**lgb_params)
# lgbm_model.fit(X,y)


cb_model = CatBoostClassifier()
# cb_model.fit(X,y)


# ## Voting Ensemble

voter = VotingClassifier(estimators=[("m1", xgb_model), ("m2", lgbm_model), ("m3", cb_model)], voting="soft",
                         weights=[0.2, 0.4, 0.4])
# voter.fit(X,y)


# # # Submission

# y_preds =voter.predict_proba(test)


# sample["Exited"] = y_preds[:,1]


# sample.to_csv("submission.csv", index=False)
