#!/usr/bin/env python
# coding: utf-8

# # Predicting Forest Cover Type

# ### Thank you for opening this kernel!

# > PROJECT CONTENT:
# 1. Import Necessary Libraries
# 2. Data Exploration/ Analysis/ Visualizing
# 3. Correlation & Correlation Matrix
# 4. Predictive Modeling
# 5. Confusion Matrix
# 6. Precision and Recall
# 7. Hyperparameters Tuning
# 8. Ensemble Methods

# > Goal:
# * The goal of this competition is to predict Forest Cover Type. We will practice Classification Algorithms to achieve the lowest prediction error.

# # Predictive Modeling:
# 1. Logistic Regression
# 2. KNN Classifier
# 3. Gaussian Naive Bayes
# 4. Support Vector Machine(SVM)
# 5. Decision Tree
# 6. Random Forest

# ### Import Necessary Libraries and Data Sets.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Import the necessary packages
import numpy as np
import pandas as pd

import warnings

warnings.simplefilter(action="ignore")

from collections import Counter

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error

# FILE_PATH = "../data/"
FILE_PATH = "./workspace/hyperopt/fstp2/data/"

submission_path = "best_submission.csv"
RANDOM_SEED = 73

# Load data and save response as target
train = pd.read_csv(FILE_PATH + "train.csv", index_col="Id")
test = pd.read_csv(FILE_PATH + "test.csv", index_col="Id")

# * Combine train and test sets

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train["Cover_Type"].to_frame()

# Combine train and test sets
concat_data = pd.concat((train, test), sort=False).reset_index(drop=True)
# Drop the target "Cover_Type" and Id columns
concat_data.drop(["Cover_Type"], axis=1, inplace=True)
concat_data.drop(["Id"], axis=1, inplace=True)
# print("Total size is :",concat_data.shape)


# concat_data.head()


# concat_data.tail()


# concat_data.info()


# ### Missing Values

# Count the null columns
null_columns = concat_data.columns[concat_data.isnull().any()]
concat_data[null_columns].isnull().sum()

# *There are no missing values in this dataset. Let"s go define numerical and categorical features.*

numeric_features = concat_data.select_dtypes(include=[np.number])
categoricals = concat_data.select_dtypes(exclude=[np.number])

# print(f"Numerical features: {numeric_features.shape}")
# print(f"Categorical features: {categoricals.shape}")


concat_data.columns

# we split the combined dataset to the original train and test sets
TrainData = concat_data[:ntrain]
TestData = concat_data[ntrain:]

# TrainData.shape, TestData.shape


# TrainData.info()


# TestData.info()


target = train[["Cover_Type"]]

# print("We make sure that both train and target sets have the same row number:")
# print(f"Train: {TrainData.shape[0]} rows")
# print(f"Target: {target.shape[0]} rows")


# Remove any duplicated column names
concat_data = concat_data.loc[:, ~concat_data.columns.duplicated()]

x = TrainData
y = np.array(target)

from sklearn.model_selection import train_test_split

# Split the data set into train and test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# > Scince we have outliers for scaling data using the mean and variance of the data is likely to not work very well. In this case, we can use robust_scale and RobustScaler as drop-in replacements instead.

scaler = RobustScaler()

# transform "x_train"
x_train = scaler.fit_transform(x_train)
# transform "x_test"
x_test = scaler.transform(x_test)
# Transform the test set
X_test = scaler.transform(TestData)

# # Building Machine Learning Models
# 1. Logistic Regression

# Baseline model of Logistic Regression with default parameters:

# Test with new parameter for KNN model
knn = KNeighborsClassifier(leaf_size=1, n_neighbors=1, p=1)
knn_mod = knn.fit(x_train, y_train)

# 5. Random Forest

random_forest = RandomForestClassifier()
random_forest_mod = random_forest.fit(x_train, y_train)

vote = VotingClassifier([("Random Forest", random_forest_mod), ("KNN Classifier", knn_mod)])
# vote_mod = vote.fit(x_train, y_train.ravel())
# vote_pred = vote_mod.predict(x_test)

# print(f"Root Mean Square Error test for ENSEMBLE METHODS: {round(np.sqrt(mean_squared_error(y_test, vote_pred)), 3)}")


# # #### Submission to Kaggle

# test["Id"].value_counts()


# Final_Submission_ForestCoverType = pd.DataFrame({
#         "Id": test["Id"],
#         "Cover_Type": vote_mod.predict(X_test)})
