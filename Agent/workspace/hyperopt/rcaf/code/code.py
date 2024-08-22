#!/usr/bin/env python
# coding: utf-8

# # Submission Notebook

# # Import Libraries

# Import numpy, pandas, and matplotlib using the standard aliases.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the following tools from sklearn: 
#     Pipeline, SimpleImputer, ColumnTransformer, OneHotEncoder, StandardScaler
#     LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split



from sklearn.model_selection import GridSearchCV
# Import joblib
import joblib
import os
import gc

# FILE_PATH="../data/"
FILE_PATH= "./workspace/hyperopt/rcaf/data/"
TARGET = "NObeyesdad"
submission_path="ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

# # Load Training Data

train = pd.read_csv(FILE_PATH+'train.csv')
train = train.sample(frac=1, random_state=1)
print(train.shape)


y_train = train.event.values
train.drop(['crew', 'experiment', 'time', 'seat', 'event'], axis=1, inplace=True)


x_train = train.iloc[:,0:27]
x_train.head()


# # Train Model

LGBM_clf = LGBMClassifier(n_estimators=150, max_depth=9, learning_rate=0.17833752251321472, boosting_type='gbdt', subsample=0.6819212428783524, random_state=1)
# LGBM_clf.fit(x_train, y_train)
# LGBM_clf.score(x_train, y_train)


# # # Test

# get_ipython().run_cell_magic('time', '', "\ncs = 1000000\ni = 0\n\nfor test in pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=cs):\n    \n    print('--Iteration',i, 'is started')\n    \n    test_pred = LGBM_clf.predict_proba(test.iloc[:,5:])\n    \n    partial_submission = pd.DataFrame({\n        'id':test.id,\n        'A':test_pred[:,0],\n        'B':test_pred[:,1],\n        'C':test_pred[:,2],\n        'D':test_pred[:,3]\n    })\n    \n    if i == 0:\n        submission = partial_submission.copy()\n    else:\n        submission = submission.append(partial_submission, ignore_index=True)\n        \n    del test\n    print('++Iteration', i, 'is done!')\n    i +=1\n")


# plt.figure(figsize=[8,4])
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.hist(submission.iloc[:,i+1], bins=20,edgecolor='k')
# plt.tight_layout()
# plt.show()


# submission.head()


# submission.to_csv("submission.csv", index=False)

