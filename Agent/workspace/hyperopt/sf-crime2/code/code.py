#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# FILE_PATH= "../data/"
FILE_PATH = "./workspace/hyperopt/sf-crime2/data/"

submission_path = "ori_submission.csv"
RANDOM_SEED = 73

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won"t be saved outside of the current session


# # Insights

# Lets start by visualising the datasets in pandas

test_set_full_raw = pd.read_csv(FILE_PATH + "test.csv.zip", compression="zip", index_col="Id")
train_set_full_raw = pd.read_csv(FILE_PATH + "train.csv.zip", compression="zip")

# train_set_full_raw.head()


# #Check what columns are present in the test set.
# train_set_full_raw.columns


# # We will check if the train set contains any null values or missing rows. The data looks fine here so no data cleaning needed.

# train_set_full_raw.info()


# train_set_full_raw.describe()


# However, looking at the summary of X, Y columns, there seems to be an outlier. Most of the Y falls around the 37 range but there is one at 90. Lets remove the 90 from our dataset

ts = train_set_full_raw.copy()
ts = ts[ts["Y"] < 90]

# By plotting the X, Y coordinates we can see that the crimes cluster at certain regions of SF.

ts.plot(x="X", y="Y", kind="scatter", alpha=0.01, figsize=(15, 12))

# We can also see the most common crime categories recorded and what are their most common resolutions

ts["Category"].value_counts()

# print("Most common resolutions for each category in percentage\n")
# for i in ts.groupby(["Category"])["Resolution"]:
#   print("\033[95m"+i[0]+"\033[0m")
#   print(round(i[1].value_counts()[:3]/i[1].count()*100,1))
#   print()


# The dataset gave us the dates of the crime. However, a more useful feature would be the hour the crime occured. Therefore, lets add it to our dataset.

from datetime import datetime

ts["Hour"] = ts.Dates.apply(lambda date_string: date_string[11:-6])
ts.head()

# Lets see which day of the week and hour crime occurs the most:

ts.groupby(["DayOfWeek"])["Hour"].value_counts()

# # Preprocessing

# Now we can start preparing our dataset for training. We will only extract the relevant columns into our train and test set.

import numpy as np

train_full = ts.copy()
X_train_full, y_train_full = np.array(train_full[["DayOfWeek", "PdDistrict", "X", "Y", "Hour"]]), np.array(
    train_full[["Category"]])
y_train_full = y_train_full.ravel()
X_test = test_set_full_raw.copy()
X_test["Hour"] = X_test.Dates.apply(lambda date_string: date_string[11:-6])
X_test = X_test.drop(columns=["Dates", "Address"])
X_test = np.array(X_test)

# We further split the full training set into train and validation set. We use Stratified sampling to ensure that the training and val set contains a proper representation of the categories present in the total population

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for train_index, test_index in sss.split(X_train_full, y_train_full):
    X_train, y_train = X_train_full[train_index], y_train_full[train_index]
    X_val, y_val = X_train_full[test_index], y_train_full[test_index]

# Next, lets set a pipeline to preprocess the datasets. StandardScaler() to normalise the numerical atttributes and OneHotEncoder() to convert the categorical attributes to arrays.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_attribs = [2, 3]
cat_attribs = [0, 1, 4]

num_pipeline = Pipeline([
    ("std_scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_val_prepared = full_pipeline.transform(X_val)
X_test_prepared = full_pipeline.transform(X_test)

# # Select and train model

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost

X_train_prepared.shape

# The train set contains 658 486 rows which lead to slow training time for me. Therefore, we will reduce the training set to 100 000 rows use Stratified Sampling again to get a good representation of the population.

ss = StratifiedShuffleSplit(n_splits=1, train_size=100_000, random_state=42)
for train_index, _ in ss.split(X_train_prepared, y_train):
    X_train_prepared_small, y_train_small = X_train_prepared[train_index], y_train[train_index].ravel()

X_train_prepared_small.shape, y_train_small.shape

# Ensemble Learning aggregates the prediction of a group of predictors. We usually get better results from it compared with just a single best individual predictor.
# 
# From a prior, not very thorough, testing with different classifiers such as LinearSVC, BaggingClassifier, ExtraTreesClassifier. I found that Ensemble learning with XGBoost and RandomForest yield the best results.

rf_clf = RandomForestClassifier(max_depth=16, random_state=42, n_jobs=-1, verbose=3)
xg_clf = xgboost.XGBClassifier()

estimators = [
    ("rf", rf_clf),
    ("xg", xg_clf)
]

voting_clf = VotingClassifier(estimators, n_jobs=-1, voting="soft")
# voting_clf.fit(X_train_prepared_small, y_train_small)
# voting_clf.score(X_val_prepared, y_val)


# # Finally, lets create the csv file for submission. This submission should give us a score of about 2.506:

# y_pred = voting_clf.predict_proba(X_test_prepared)
# pred_df = pd.DataFrame(y_pred, columns=[voting_clf.classes_])
# pred_df["Id"]= list(range(pred_df.shape[0]))
# pred_df.to_csv("crime_pred_02.zip", compression="zip", index=False)
# pred_df.head()
