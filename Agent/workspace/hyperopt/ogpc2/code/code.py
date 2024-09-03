#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import xgboost as xgb

from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

FILE_PATH = "./workspace/hyperopt/ogpc2/data/"

# Any results you write to the current directory are saved as output.


train = pd.read_csv(FILE_PATH+"train.csv.zip")
test = pd.read_csv(FILE_PATH+"test.csv.zip")


sample = pd.read_csv(FILE_PATH+'sampleSubmission.csv.zip')


train.head()


target = train['target']
train.drop(['id','target'],axis=1,inplace=True)
train.shape


testId = test['id']
test.drop('id',axis=1,inplace=True)


rfc = RandomForestClassifier(n_estimators=50,random_state=1412,n_jobs=-1)
# rfc.fit(train,target)
# preds = rfc.predict_proba(test)


lr = LogisticRegression()
# lr.fit(train,target)
# lpreds = lr.predict_proba(test)


xg = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
# xg.fit(train,target)
# xpreds = xg.predict_proba(test)


# finalPreds = 0.33*preds+0.33*lpreds+0.33*xpreds
# finalPreds = lpreds


# pred = pd.DataFrame(finalPreds, index=sample.id.values, columns=sample.columns[1:])
# pred.to_csv('onlylr.csv', index_label='id')

from sklearn.ensemble import VotingClassifier

voting_classifier=VotingClassifier(estimators=[('rfc',rfc),('lr',lr),('xg',xg)],voting='soft')

