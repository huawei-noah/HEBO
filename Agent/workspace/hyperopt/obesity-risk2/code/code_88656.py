#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
###for the model and evaluation###
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


FILE_PATH= "./workspace/hyperopt/obesity-risk/data/"
# FILE_PATH= "../data/"

TARGET = "NObeyesdad"
submission_path="ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

import os
df_train = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(FILE_PATH, "test.csv"))


print("\n")
print(f'Train lenght: {df_train.shape[0]}')
print(f'Test lenght: {df_test.shape[0]}')

# Features related wih eating habits:
# 
# * FAVC: Frequent consumption of high caloric food;
# * FCVC: Frequency of consumption of vegetables;
# * NCP: Number of main meals;
# * CAEC: Consumption of food between meals;
# * CH20: Consumption of water daily;
# * CALC: Consumption of alcohol.
# 
# Features related with the physical condition:
# 
# * SCC: Calories consumption monitoring
# * FAF: Physical activity frequency;
# * TUE: Time using technology devices;
# * MTRANS: Transportation used.

#EDA on Data
df_train.NObeyesdad.value_counts()

df_train.info()

# It is a Multiclass Classification problem

# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows  
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column . 
df_train['Gender']= label_encoder.fit_transform(df_train['Gender']) 
df_train['family_history_with_overweight']= label_encoder.fit_transform(df_train['family_history_with_overweight']) 
df_train['FAVC']= label_encoder.fit_transform(df_train['FAVC']) 
df_train['CAEC']= label_encoder.fit_transform(df_train['CAEC']) 
df_train['SMOKE']= label_encoder.fit_transform(df_train['SMOKE']) 
df_train['SCC']= label_encoder.fit_transform(df_train['SCC']) 
df_train['CALC']= label_encoder.fit_transform(df_train['CALC']) 
df_train['MTRANS']= label_encoder.fit_transform(df_train['MTRANS']) 
#df_train['NObeyesdad']= label_encoder.fit_transform(df_train['NObeyesdad']) 


df_train.tail()

df_corr=df_train.drop(['id','NObeyesdad'], axis=1)

import seaborn as sns

# load the Auto dataset
auto_df = sns.load_dataset('mpg')

# calculate the correlation matrix on the numeric columns
corr = df_corr.select_dtypes('number').corr()

# plot the heatmap
# sns.heatmap(corr)


from sklearn.ensemble import RandomForestClassifier

y = df_train["NObeyesdad"]

features = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
X = df_train[features]
X_val = df_test[features]

from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Encode labels in column . 
X_val['Gender']= label_encoder.fit_transform(X_val['Gender']) 
X_val['family_history_with_overweight']= label_encoder.fit_transform(X_val['family_history_with_overweight']) 
X_val['FAVC']= label_encoder.fit_transform(X_val['FAVC']) 
X_val['CAEC']= label_encoder.fit_transform(X_val['CAEC']) 
X_val['SMOKE']= label_encoder.fit_transform(X_val['SMOKE']) 
X_val['SCC']= label_encoder.fit_transform(X_val['SCC']) 
X_val['CALC']= label_encoder.fit_transform(X_val['CALC']) 
X_val['MTRANS']= label_encoder.fit_transform(X_val['MTRANS']) 


predictions = model.predict(X_val)

output = pd.DataFrame({ 'Survived': predictions})


submission_df=pd.merge(df_test['id'],output , left_index=True, right_index=True)
submission_df.head()

submission_df.to_csv(os.path.join(FILE_PATH,submission_path),index=False)
print("Random Forest Submission")


score=1-accuracy


