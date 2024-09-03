#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

FILE_PATH = "./workspace/hyperopt/abalone/data/"
# FILE_PATH="../data/"
submmision_path = "best_submission.csv"
RANDOM_SEED = 42

feature_importances_path = "feature_importances.csv"
selected_features_path = "selected_features.csv"

# Load the data
df_train = pd.read_csv(FILE_PATH + "train.csv", index_col="id")
df_test = pd.read_csv(FILE_PATH + "test.csv", index_col="id")

# Data Summary Statistics
# print("Train Data Description:")
# print(df_train.describe())

X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]

# Convert "sex" column to category
sex_mapping = {"M": 0, "F": 1, "I": 2}
X["Sex"] = X["Sex"].map(sex_mapping)
df_test["Sex"] = df_test["Sex"].map(sex_mapping)

# Feature engineering

# # Step 1: Train the model
# model = LGBMRegressor(verbose=0, random_state=1) #dont add any parameters to this model
# model.fit(X, y)

# # Step 2: Get feature importances
# importances = model.feature_importances_

# # Create a DataFrame for better handling
# feature_importances = pd.DataFrame({"feature": X.columns, "importance": importances})

# # Step 3: Sort features by importance
# feature_importances = feature_importances.sort_values(by="importance", ascending=True)

# # # Print feature importances
# # print("\nFeature Importances:")
# # print(feature_importances)


# # Save feature importances
# feature_importances.to_csv(feature_importances_path, index=False)


# Load feature importances
# feature_importances = pd.read_csv(FILE_PATH+feature_importances_path)


# kf = KFold(n_splits=7, shuffle=True, random_state=1)

# best_i = 0
# best_score = 1
# for i in range(3):
#     # Step 5: Select features that contribute to the top N% of total importance
#     selected_features = feature_importances[i:]

#     # Extract the column names of the selected features
#     selected_columns = selected_features["feature"]

#     # Create the new DataFrame with only the selected features
#     X2 = X[selected_columns]
#     scores = cross_val_score(model, X2, y, scoring="neg_mean_squared_log_error", cv=kf, n_jobs=-1)
#     mean_score = -np.mean(scores)
#     RMLSE = np.sqrt(mean_score)
#     print(f"RMSLE for removing {i} least important features: {RMLSE}")

#     if RMLSE < best_score:
#         best_score = RMLSE
#         best_i = i

# print(f"\nBest RMLSE is {best_score} and {best_i} least important features were removed\n")
# # Remove extra features
# selected_features = feature_importances[best_i:]
# selected_features.to_csv(FILE_PATH+selected_features_path)

selected_features = pd.read_csv(FILE_PATH + selected_features_path)
X = X[selected_features["feature"]]
df_test = df_test[selected_features["feature"]]

# Modeling and Evaluation
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=RANDOM_SEED)
# CatBoost columns
# cat_columns = X.select_dtypes(include="category").columns.tolist()
# print(cat_columns) # ["Sex"]

# List of models to evaluate
catboost_model = CatBoostRegressor(random_state=1, verbose=False)


# # Fit the models on the training data
# catboost_model.fit(X_train, y_train)


# # Evaluate the models
# catboost_preds = catboost_model.predict(X_val)


# final_preds = np.round((catboost_preds + lgbm_preds + xgb_preds) / 3).astype("int")

# final_score = mean_squared_log_error(y_val, final_preds)

# # Make predictions on the test set
# catboost_preds = catboost_model.predict(df_test)
# lgbm_preds = lgbm_model.predict(df_test)
# xgb_preds = xgb_model.predict(df_test)

# # Averaging predictions
# final_preds = np.round((catboost_preds + lgbm_preds + xgb_preds) / 3).astype("int")

# # Submission
# submission = pd.DataFrame({"id": df_test.index, "Rings": final_preds})
# submission.to_csv(submmision_path, header=True, index=False)

# score= final_score
