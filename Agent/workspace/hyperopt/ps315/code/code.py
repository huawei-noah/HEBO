#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

FILE_PATH = "./workspace/hyperopt/ps315/data/"    

data = pd.read_csv(FILE_PATH+'data.csv.zip')


# ProfileReport(data)


# data.columns


data.columns = ['id', 'author', 'geometry', 'pressure', 'mass_flux', 'x_e_out', 'D_e', 'D_h', 'length', 'chf_exp']


# data


train = data[~data.x_e_out.isna()]
test = data[data.x_e_out.isna()]


# train.describe()


# test.describe()


# ### Sanity check

# tmp = data.drop(['id', 'x_e_out'], axis=1)
# tmp_target = data.x_e_out.isna() * 1.0


# X_tmp_train, X_tmp_test, y_tmp_train, y_tmp_test = train_test_split(tmp, tmp_target)


# X_tmp_train.isna().sum()


# fill_num = X_tmp_train.median()


# X_tmp_train.fillna(fill_num).isna().sum()


# X_tmp_train['author'].value_counts()


# X_tmp_train['geometry'].value_counts()


# X_tmp_train = X_tmp_train.fillna(fill_num)
# X_tmp_test = X_tmp_test.fillna(fill_num)


# X_tmp_train[['author', 'geometry']] = X_tmp_train[['author', 'geometry']].fillna({'author': 'Thompson', 'geometry': 'tube'})
# X_tmp_test[['author', 'geometry']] = X_tmp_test[['author', 'geometry']].fillna({'author': 'Thompson', 'geometry': 'tube'})


# X_tmp_train


# geometry_encoder = pd.DataFrame({
#     'geometry': X_tmp_train.geometry,
#     'target': y_tmp_train
# }).groupby('geometry').target.mean()

# author_encoder = pd.DataFrame({
#     'author': X_tmp_train.author,
#     'target': y_tmp_train
# }).groupby('author').target.mean()


# X_tmp_train.author


# X_tmp_train.author = X_tmp_train.author.map(author_encoder)
# X_tmp_train.geometry = X_tmp_train.geometry.map(geometry_encoder)

# X_tmp_test.author = X_tmp_test.author.map(author_encoder)
# X_tmp_test.geometry = X_tmp_test.geometry.map(geometry_encoder)


# X_tmp_train


# clf = RandomForestClassifier(max_depth=3)
# clf.fit(X_tmp_train, y_tmp_train)


# print(clf.score(X_tmp_train, y_tmp_train))
# print(clf.score(X_tmp_test, y_tmp_test))


# y_tmp_train.mean()


# 1-y_tmp_train.mean()


# train_proba = clf.predict_proba(X_tmp_train)[:,1]
# train_proba.mean(), train_proba.std()


# roc_auc_score(y_tmp_train, train_proba)


# roc_auc_score(y_tmp_test, clf.predict_proba(X_tmp_test)[:,1])


# ### Conclusion
# Train and test are not different from each other.
# They came out of the same general distribution.
# We can say that the omissions occurred by accident.

# ## Train the main model
X_train=train.drop(['id', 'x_e_out'],axis=1)
y_train=train.x_e_out
# X_train, X_val, y_train, y_val = train_test_split(train.drop(['id', 'x_e_out'],axis=1), train.x_e_out)


# Fix NA

numeric_cols = X_train.select_dtypes(include='number').columns
non_numeric_cols = X_train.select_dtypes(exclude='number').columns

# Compute median for numeric columns
fill_num = X_train[numeric_cols].median()

# Fill missing values in numeric columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(fill_num)

# Fill missing values in non-numeric columns
X_train[['author', 'geometry']] = X_train[['author', 'geometry']].fillna({'author': 'Thompson', 'geometry': 'tube'})
# X_val[['author', 'geometry']] = X_val[['author', 'geometry']].fillna({'author': 'Thompson', 'geometry': 'tube'})


geometry_encoder = pd.DataFrame({
    'geometry': X_train.geometry,
    'target': y_train
}).groupby('geometry').target.mean()

author_encoder = pd.DataFrame({
    'author': X_train.author,
    'target': y_train
}).groupby('author').target.mean()

X_train.author = X_train.author.map(author_encoder)
X_train.geometry = X_train.geometry.map(geometry_encoder)

# X_val.author = X_val.author.map(author_encoder)
# X_val.geometry = X_val.geometry.map(geometry_encoder)


# X_train.isna().sum()


# X_val.isna().sum()

from sklearn.metrics import mean_squared_error
model = RandomForestRegressor(max_depth=5)
# model.fit(X_train, y_train)


# np.mean((model.predict(X_train) - y_train)**2)**0.5


# np.mean((model.predict(X_val) - y_val)**2)**0.5


# X_test = test.drop(['id', 'x_e_out'],axis=1)
# X_test[['author', 'geometry']] = X_test[['author', 'geometry']].fillna({'author': 'Thompson', 'geometry': 'tube'})
# X_test = X_test.fillna(fill_num)
# X_test.author = X_test.author.map(author_encoder)
# X_test.geometry = X_test.geometry.map(geometry_encoder)


# model.predict(X_test)


# submit = pd.read_csv('/kaggle/input/playground-series-s3e15/sample_submission.csv')


# submit['x_e_out [-]'] = model.predict(X_test)


# submit.to_csv('random_forest.csv', index=False)

