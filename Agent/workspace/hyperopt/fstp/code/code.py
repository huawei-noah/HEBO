#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won"t be saved outside of the current session


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# FILE_PATH="../data/"
FILE_PATH = "./workspace/hyperopt/fstp/data/"
TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
n_splits = 9
RANDOM_SEED = 73

# Load the datasets
train_data = pd.read_csv(FILE_PATH + "train.csv")
test_data = pd.read_csv(FILE_PATH + "test.csv")

# Define features and target variable
X = train_data.drop(["Id", "Cover_Type"], axis=1)
y = train_data["Cover_Type"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# # Fit the model on the training data
# model.fit(X_train, y_train)


# # Make predictions on the validation set
# val_predictions = model.predict(X_val)


# # Calculate accuracy score on the validation set
# accuracy = accuracy_score(y_val, val_predictions)


# # Once you are satisfied with the model performance, you can make predictions on the test set
# test_features = test_data.drop("Id", axis=1)
# test_predictions = model.predict(test_features)


# # Create a submission file
# submission = pd.DataFrame({"Id": test_data["Id"], "Cover_Type": test_predictions})
# submission.to_csv(submission_path, index=False)


# score=1.0-accuracy
