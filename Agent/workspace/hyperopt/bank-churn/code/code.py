#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# FILE_PATH= "../data/"
FILE_PATH = "./workspace/hyperopt/bank-churn/data/"

TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
n_splits = 9
RANDOM_SEED = 73


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won"t be saved outside of the current session

def OHE(x_train, x_test, test, column):
    encoder = OneHotEncoder(handle_unknown="ignore")

    # Fit and transform the training data, then transform the test data
    encoder_x_train = encoder.fit_transform(x_train[[column]]).toarray()
    encoder_x_test = encoder.transform(x_test[[column]]).toarray()
    encoder_test = encoder.transform(test[[column]]).toarray()

    # Create DataFrames using the dense arrays
    encoder_x_train_df = pd.DataFrame(encoder_x_train, columns=encoder.get_feature_names_out([column]))
    encoder_x_test_df = pd.DataFrame(encoder_x_test, columns=encoder.get_feature_names_out([column]))
    encoder_test_df = pd.DataFrame(encoder_test, columns=encoder.get_feature_names_out([column]))

    # Concatenate the original dataframes with the new one-hot encoded columns
    x_train = pd.concat([x_train.drop(column, axis=1).reset_index(drop=True), encoder_x_train_df], axis=1)
    x_test = pd.concat([x_test.drop(column, axis=1).reset_index(drop=True), encoder_x_test_df], axis=1)
    test = pd.concat([test.drop(column, axis=1).reset_index(drop=True), encoder_test_df], axis=1)

    return x_train, x_test, test


data = pd.read_csv(FILE_PATH + "train.csv")
test = pd.read_csv(FILE_PATH + "test.csv")

data.sample(5)

data = data.drop(columns=["id", "CustomerId", "Surname"], axis=1)
test = test.drop(columns=["id", "CustomerId", "Surname"], axis=1)

data.info()

data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
test["Gender"] = test["Gender"].map({"Male": 0, "Female": 1})

x = data.drop("Exited", axis=1)
y = data["Exited"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

x_train["Geography"].value_counts()

x_train, x_test, test = OHE(x_train, x_test, test, "Geography")

models = {
    #     "Logistic Regression": LogisticRegression(),
    # #     "Support Vector Classifier": SVC(),
    #     "Decision Tree Classifier": DecisionTreeClassifier(),
    #     "Random Forest Classifier": RandomForestClassifier(),
    #     "Gradient Boosting Classifier": GradientBoostingClassifier(),
    #     "XGBoost Classifier": XGBClassifier(),
    #     "LightGBM Classifier": LGBMClassifier(),
    "CatBoost Classifier": CatBoostClassifier(silent=True),
    # "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
    # "AdaBoost Classifier": AdaBoostClassifier(),
    # "Naive Bayes Classifier": GaussianNB()
}
model = models["CatBoost Classifier"]
# for name , model in models.items():
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name}: Accuracy - {accuracy:.4f}")

# sample_submission_df = pd.read_csv(FILE_PATH+"sample_submission.csv")
# sample_submission_df["Exited"] = models["CatBoost Classifier"].predict(test)
# sample_submission_df.to_csv(submission_path, index=False)
# sample_submission_df.head()
# score=1.0 - score
