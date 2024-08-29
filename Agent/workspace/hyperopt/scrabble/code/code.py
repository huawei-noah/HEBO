#!/usr/bin/env python
# coding: utf-8

# # <h><center>⭐️⭐️Scrabble Player Rating(Predict players" ratings based on Woogles.io gameplay)⭐️⭐️</center></h>
# <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn.vox-cdn.com%2Fthumbor%2F68c3cH43YrqhhjJOWczTlra59a8%3D%2F0x34%3A640x394%2F1600x900%2Fcdn.vox-cdn.com%2Fuploads%2Fchorus_image%2Fimage%2F29923105%2F937649420_d471237362_z.0.jpg&f=1&nofb=1&ipt=16351708cb943c73ec8fc044fe9657fc25b97caa638d15e744f0753a5bfb1c7f&ipo=images">
# 
# ## ***This Notebook is ordinary baseline approach only train, test ,sample csv file used and applied preprocessing, Build the XGBoost regression method.***
# 
# ## **⭐️Try it yourself guys, simple baseline approach⭐️***

# # **Import Necessary library**

# Import necessay libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn import model_selection
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# # **Load Data file**

FILE_PATH = "../data/"
FILE_PATH= "./workspace/hyperopt/scrabble/data/"
# TARGET = "NObeyesdad"
submission_path = "ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

# import the data and shape
train = pd.read_csv(FILE_PATH + "train.csv")
test = pd.read_csv(FILE_PATH + "test.csv")
sample = pd.read_csv(FILE_PATH + "sample_submission.csv")
games = pd.read_csv(FILE_PATH + "games.csv")
turns = pd.read_csv(FILE_PATH + "turns.csv")

# print(train.shape, test.shape, sample.shape, games.shape, turns.shape)

# # **KFold**

# add extra one columns
train["kfold"] = -1
# Distributing the data 5 shares
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_indicies, valid_indicies) in enumerate(kfold.split(X=train)):
    # print(fold,train_indicies,valid_indicies)
    train.loc[valid_indicies, "kfold"] = fold

print(train.kfold.value_counts())  # total data 300000 = kfold split :5 * 60000

# output of train folds data
train.to_csv("trainfold_5.csv", index=False)

train = pd.read_csv("./trainfold_5.csv")

# # **Preprocessing**

# train.isnull().sum()


# test.isnull().sum()


test = test.drop("rating", axis=1)

train["nickname"] = train["nickname"].astype("category").cat.codes

test["nickname"] = test["nickname"].astype("category").cat.codes

# # **Feature spliting**

X = train.drop(["rating", "kfold"], axis=1)
y = train["rating"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# # **Build the Model**

# store the final_prediction data and score
final_predictions = []
scores = []

xgb_params = {
    "learning_rate": 0.0011896,
    "subsample": 0.7875490025178,
    "colsample_bytree": 0.11807135201147,
    "max_depth": 6,
    "booster": "gbtree",
    "reg_lambda": 0.0008746338866473539,
    "reg_alpha": 23.13181079976304,
    "random_state": 40,
    "n_estimators": 30000
}

model = XGBRegressor(**xgb_params,
                     tree_method="gpu_hist",
                     predictor="gpu_predictor",
                     device="gpu")
# for fold in range(5):

#     #Model hyperparameter of XGboostRegressor
#     xgb_params = {
#         "learning_rate": 0.0011896,
#         "subsample": 0.7875490025178,
#         "colsample_bytree": 0.11807135201147,
#         "max_depth": 6,
#         "booster": "gbtree", 
#         "reg_lambda": 0.0008746338866473539,
#         "reg_alpha": 23.13181079976304,
#         "random_state":40,
#         "n_estimators":30000
#     }

#     model= XGBRegressor(**xgb_params,
#                        tree_method="gpu_hist",
#                        predictor="gpu_predictor",
#                        device="gpu")
#     model.fit(x_train,y_train,early_stopping_rounds=300,eval_set=[(x_test,y_test)],verbose=2000)
#     preds_valid = model.predict(x_test)

#     #Training model apply the test data and predict the output
#     test_pre = model.predict(test)
#     final_predictions.append(test_pre)

#     #Rootmeansquared output
#     rms = mean_squared_error(y_test,preds_valid,squared=False)

#     scores.append(rms)
#     #way of output is display
#     print(f"fold:{fold},rmse:{rms}")


# # **Prediction and generate submission csv file**

# # prediction of data
# preds = np.mean(np.column_stack(final_predictions),axis=1)
# print(preds)
# sample.rating = preds[0:22363]
# sample.to_csv("submission.csv",index=False)
# print("success")


# ## **Thanks for visiting guys**


# score=np.mean(scores)
