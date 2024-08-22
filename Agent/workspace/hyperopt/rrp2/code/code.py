import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# FILE_PATH="../data/"
FILE_PATH = "./workspace/hyperopt/rrp/data/"
TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
n_splits = 9
RANDOM_SEED = 73

df = pd.read_csv(FILE_PATH + "train.csv.zip")
df.shape

# The dataset is quite small so complex models with many parameters should be avoided. Using a complex model for this dataset will cause the model to overfit to the dataset. Regularization techniques will definitely need to be used to prevent the possibility of overfitting.

test_df = pd.read_csv(FILE_PATH + "test.csv.zip")
test_df.shape

# The **MB** Type will be replaced with the **DT** Type in the test set since it"s not available in our training set. The **City** feature is useless since our training set contains **34** unique cities but the test set contains **57** unique cities.

test_df.loc[test_df["Type"] == "MB", "Type"] = "DT"

df.drop("City", axis=1, inplace=True)
test_df.drop("City", axis=1, inplace=True)

import datetime

df.drop("Id", axis=1, inplace=True)
df["Open Date"] = pd.to_datetime(df["Open Date"])
test_df["Open Date"] = pd.to_datetime(test_df["Open Date"])
launch_date = datetime.datetime(2015, 3, 23)
# scale days open
df["Days Open"] = (launch_date - df["Open Date"]).dt.days / 1000
test_df["Days Open"] = (launch_date - test_df["Open Date"]).dt.days / 1000
df.drop("Open Date", axis=1, inplace=True)
test_df.drop("Open Date", axis=1, inplace=True)

# # Feature Engineering
# 
# If the distribution of the data is left skewed, the skewness values will be negative. If the distribution of the data is right skewed, the skewness values will be positive.

# copy_df = df.copy()
# copy_test_df = test_df.copy()
# numeric_features = df.dtypes[df.dtypes != "object"].index
# skewed_features = df[numeric_features].apply(lambda x: skew(x))
# skewed_features = skewed_features[skewed_features > 0.5].index
# df[skewed_features] = np.log1p(df[skewed_features])
# test_df[skewed_features.drop("revenue")] = np.log1p(test_df[skewed_features.drop("revenue")])
# Above handles skewed features using log transformation
# Below uses multiple imputation for P1-P37, since they are actually categorical
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp_train = IterativeImputer(max_iter=30, missing_values=0, sample_posterior=True, min_value=1, random_state=37)
imp_test = IterativeImputer(max_iter=30, missing_values=0, sample_posterior=True, min_value=1, random_state=23)

p_data = ["P" + str(i) for i in range(1, 38)]
df[p_data] = np.round(imp_train.fit_transform(df[p_data]))
test_df[p_data] = np.round(imp_test.fit_transform(test_df[p_data]))

# drop_first=True for Dummy Encoding for object types, and drop_first=False for OHE
columnsToEncode = df.select_dtypes(include=[object]).columns
df = pd.get_dummies(df, columns=columnsToEncode, drop_first=False)
test_df = pd.get_dummies(test_df, columns=columnsToEncode, drop_first=False)

df["revenue"] = np.log1p(df["revenue"])
X, y = df.drop("revenue", axis=1), df["revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)

# # Ridge and Lasso Regression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

params_ridge = {
    "alpha": [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],
    "fit_intercept": [True, False],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
}

ridge_model = Ridge()
ridge_regressor = GridSearchCV(ridge_model, params_ridge, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
ridge_regressor.fit(X_train, y_train)

ridge_model = Ridge(alpha=ridge_regressor.best_params_["alpha"],
                    fit_intercept=ridge_regressor.best_params_["fit_intercept"],
                    solver=ridge_regressor.best_params_["solver"])
# ridge_model.fit(X_train, y_train)
# y_train_pred = ridge_model.predict(X_train)
# y_pred = ridge_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# Ridge Model Feature Importance


params_lasso = {
    "alpha": [.01, .1, .5, .7, .9, .95, .99, 1, 5, 10, 20],
    "fit_intercept": [True, False],
}

lasso_model = Lasso()
lasso_regressor = GridSearchCV(lasso_model, params_lasso, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
lasso_regressor.fit(X_train, y_train)
# print(f"Optimal alpha: {lasso_regressor.best_params_["alpha"]:.2f}")
# print(f"Optimal fit_intercept: {lasso_regressor.best_params_["fit_intercept"]}")
# print(f"Optimal normalize: {lasso_regressor.best_params_["normalize"]}")
# print(f"Best score: {lasso_regressor.best_score_}")


lasso_model = Lasso(alpha=lasso_regressor.best_params_["alpha"],
                    fit_intercept=lasso_regressor.best_params_["fit_intercept"], )
# lasso_model.fit(X_train, y_train)
# y_train_pred = lasso_model.predict(X_train)
# y_pred = lasso_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# Lasso Model Feature Importance


# # ElasticNet (combination of Ridge & Lasso)

# from sklearn.linear_model import ElasticNetCV, ElasticNet

# # Use ElasticNetCV to tune alpha automatically instead of redundantly using ElasticNet and GridSearchCV
# el_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=5e-2, cv=10, n_jobs=-1)         
# el_model.fit(X_train, y_train)
# print(f"Optimal alpha: {el_model.alpha_:.6f}")
# print(f"Optimal l1_ratio: {el_model.l1_ratio_:.3f}")
# print(f"Number of iterations {el_model.n_iter_}")


# y_train_pred = el_model.predict(X_train)
# y_pred = el_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# ElasticNet Model Feature Importance
# el_feature_coef = pd.Series(index = X_train.columns, data = np.abs(el_model.coef_))
# n_features = (el_feature_coef>0).sum()
# print(f"{n_features} features with reduction of {(1-n_features/len(el_feature_coef))*100:2.2f}%")
# el_feature_coef.sort_values().plot(kind = "bar", figsize = (13,5));


# # # K-Nearest Neighbors

# from sklearn.neighbors import KNeighborsRegressor

# params_knn = {
#     "n_neighbors" : [3, 5, 7, 9, 11],
# }

# knn_model = KNeighborsRegressor()
# knn_regressor = GridSearchCV(knn_model, params_knn, scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
# knn_regressor.fit(X_train, y_train)
# print(f"Optimal neighbors: {knn_regressor.best_params_["n_neighbors"]}")
# print(f"Best score: {knn_regressor.best_score_}")


# knn_model = KNeighborsRegressor(n_neighbors=knn_regressor.best_params_["n_neighbors"])
# knn_model.fit(X_train, y_train)
# y_train_pred = knn_model.predict(X_train)
# y_pred = knn_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# # Random Forest

from sklearn.ensemble import RandomForestRegressor

params_rf = {
    "max_depth": [10, 30, 35, 50, 65, 75, 100],
    "max_features": [.3, .4, .5, .6],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [30, 50, 100, 200]
}

rf = RandomForestRegressor()
rf_regressor = GridSearchCV(rf, params_rf, scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)
rf_regressor.fit(X_train, y_train)

rf_model = RandomForestRegressor(max_depth=rf_regressor.best_params_["max_depth"],
                                 max_features=rf_regressor.best_params_["max_features"],
                                 min_samples_leaf=rf_regressor.best_params_["min_samples_leaf"],
                                 min_samples_split=rf_regressor.best_params_["min_samples_split"],
                                 n_estimators=rf_regressor.best_params_["n_estimators"],
                                 n_jobs=-1, oob_score=True)
# rf_model.fit(X_train, y_train)
# y_train_pred = rf_model.predict(X_train)
# y_pred = rf_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# # Random Forest Model Feature Importance
# rf_feature_importance = pd.Series(index = X_train.columns, data = np.abs(rf_model.feature_importances_))
# n_features = (rf_feature_importance>0).sum()
# print(f"{n_features} features with reduction of {(1-n_features/len(rf_feature_importance))*100:2.2f}%")
# rf_feature_importance.sort_values().plot(kind = "bar", figsize = (13,5));


# # # Light GBM

# import lightgbm as lgbm

# params_lgbm = {
#     "learning_rate": [.01, .1, .5, .7, .9, .95, .99, 1],
#     "boosting": ["gbdt"],
#     "metric": ["l1"],
#     "feature_fraction": [.3, .4, .5, 1],
#     "num_leaves": [20],
#     "min_data": [10],
#     "max_depth": [10],
#     "n_estimators": [10, 30, 50, 100]
# }

# lgb = lgbm.LGBMRegressor()
# lgb_regressor = GridSearchCV(lgb, params_lgbm, scoring="neg_root_mean_squared_error", cv = 10, n_jobs = -1)
# lgb_regressor.fit(X_train, y_train)
# print(f"Optimal lr: {lgb_regressor.best_params_["learning_rate"]}")
# print(f"Optimal feature_fraction: {lgb_regressor.best_params_["feature_fraction"]}")
# print(f"Optimal n_estimators: {lgb_regressor.best_params_["n_estimators"]}")
# print(f"Best score: {lgb_regressor.best_score_}")


# lgb_model = lgbm.LGBMRegressor(learning_rate=lgb_regressor.best_params_["learning_rate"], boosting="gbdt", 
#                                metric="l1", feature_fraction=lgb_regressor.best_params_["feature_fraction"], 
#                                num_leaves=20, min_data=10, max_depth=10, 
#                                n_estimators=lgb_regressor.best_params_["n_estimators"], n_jobs=-1)
# lgb_model.fit(X_train, y_train)
# y_train_pred = lgb_model.predict(X_train)
# y_pred = lgb_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# # LightGBM Feature Importance
# lgb_feature_importance = pd.Series(index = X_train.columns, data = np.abs(lgb_model.feature_importances_))
# n_features = (lgb_feature_importance>0).sum()
# print(f"{n_features} features with reduction of {(1-n_features/len(lgb_feature_importance))*100:2.2f}%")
# lgb_feature_importance.sort_values().plot(kind = "bar", figsize = (13,5));


# # # XGBoost

# params_xgb = {
#     "learning_rate": [.1, .5, .7, .9, .95, .99, 1],
#     "colsample_bytree": [.3, .4, .5, .6],
#     "max_depth": [4],
#     "alpha": [3],
#     "subsample": [.5],
#     "n_estimators": [30, 70, 100, 200]
# }

# xgb_model = XGBRegressor()
# xgb_regressor = GridSearchCV(xgb_model, params_xgb, scoring="neg_root_mean_squared_error", cv = 10, n_jobs = -1)
# xgb_regressor.fit(X_train, y_train)
# print(f"Optimal lr: {xgb_regressor.best_params_["learning_rate"]}")
# print(f"Optimal colsample_bytree: {xgb_regressor.best_params_["colsample_bytree"]}")
# print(f"Optimal n_estimators: {xgb_regressor.best_params_["n_estimators"]}")
# print(f"Best score: {xgb_regressor.best_score_}")


# xgb_model = XGBRegressor(learning_rate=xgb_regressor.best_params_["learning_rate"], 
#                          colsample_bytree=xgb_regressor.best_params_["colsample_bytree"], 
#                          max_depth=4, alpha=3, subsample=.5, 
#                          n_estimators=xgb_regressor.best_params_["n_estimators"], n_jobs=-1)
# xgb_model.fit(X_train, y_train)
# y_train_pred = xgb_model.predict(X_train)
# y_pred = xgb_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# # XGB with early stopping
# xgb_model.fit(X_train, y_train, early_stopping_rounds=4,
#              eval_set=[(X_test, y_test)], verbose=False)
# y_train_pred = xgb_model.predict(X_train)
# y_pred = xgb_model.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


# # XGB Feature Importance, relevant features can be selected based on its score
# feature_important = xgb_model.get_booster().get_fscore()
# keys = list(feature_important.keys())
# values = list(feature_important.values())

# data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
# data.plot(kind="bar", figsize = (13,5))
# plt.show()


# # Regressor Ensembling

rf_model_en = RandomForestRegressor(max_depth=200, max_features=0.4, min_samples_leaf=3,
                                    min_samples_split=6, n_estimators=30, n_jobs=-1, oob_score=True)
# rf_model_en.fit(X_train, y_train)
# y_train_pred = rf_model_en.predict(X_train)
# y_pred = rf_model_en.predict(X_test)
# print("Train r2 score: ", r2_score(y_train_pred, y_train))
# print("Test r2 score: ", r2_score(y_test, y_pred))
# train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Train RMSE: {train_rmse:.4f}")
# print(f"Test RMSE: {test_rmse:.4f}")


from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from matplotlib import pyplot


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    base_models = list()
    base_models.append(("ridge", ridge_model))
    base_models.append(("lasso", lasso_model))
    base_models.append(("rf", rf_model_en))
    # define meta learner model
    learner = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=base_models, final_estimator=learner, cv=10)
    return model


# get a list of models to evaluate
def get_models():
    models = dict()
    models["ridge"] = ridge_model
    models["lasso"] = lasso_model
    models["rf_en"] = rf_model_en
    models["stacking"] = get_stacking()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=19)
    scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, error_score="raise")
    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()

# define the base models
base_models = list()
base_models.append(("ridge", ridge_model))
base_models.append(("lasso", lasso_model))
base_models.append(("rf1", rf_model))
# base_models.append(("rf2", rf_model_en))
# base_models.append(("rf3", RandomForestRegressor(max_depth=8, max_features=0.1, min_samples_leaf=3, 
#                                                 min_samples_split=2, n_estimators=250, n_jobs=-1, oob_score=False)))
# define meta learner model
learner = LinearRegression()
# define the stacking ensemble
stack1 = StackingRegressor(estimators=base_models, final_estimator=learner, cv=10)
# # fit the model on all available data
# stack1.fit(X_train, y_train)
# y_pred = stack1.predict(X_test)
# test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# # # Submissions

# submission = pd.DataFrame(columns=["Id","Prediction"])
# submission["Id"] = test_df["Id"]


# stack_pred1 = stack1.predict(test_df.drop("Id", axis=1))
# submission["Prediction"] = np.expm1(stack_pred1)
# submission.to_csv(submission_path,index=False)

# score=test_rmse
