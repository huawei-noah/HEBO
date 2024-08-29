#!/usr/bin/env python
# coding: utf-8

# In this notebook, you will learn how to make your first submission to the [Tabular Playground Series - Feb 2021 competition.](https://www.kaggle.com/c/tabular-playground-series-feb-2021)
# 
# # Make the most of this notebook!
# 
# You can use the "Copy and Edit" button in the upper right of the page to create your own copy of this notebook and experiment with different models. You can run it as is and then see if you can make improvements.

import numpy as np
import pandas as pd
from pathlib import Path

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
        
FILE_PATH = "./workspace/hyperopt/tspf/data/"

# # Read in the data files

train = pd.read_csv(FILE_PATH + 'train.csv.zip', index_col='id')
# display(train.head())


test = pd.read_csv(FILE_PATH + 'test.csv.zip', index_col='id')
# display(test.head())


submission = pd.read_csv(FILE_PATH + 'sample_submission.csv.zip', index_col='id')
# display(submission.head())


# ## We need to encode the categoricals.
# 
# There are different strategies to accomplish this, and different approaches will have different performance when using different algorithms. For this starter notebook, we'll use simple encoding.

for c in train.columns:
    if train[c].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(train[c].values)
        test[c] = lbl.transform(test[c].values)
        
# display(train.head())


# ## Pull out the target, and make a validation split

target = train.pop('target')
X_train, X_test, y_train, y_test = train_test_split(train, target, train_size=0.60)


# # How well can we do with a completely naive model?
# 
# We'll want any of our models to do (hopefully much!) better than this.

# Let's get a benchmark score

# print(f'{score_dummy:0.5f}')


# # Simple Linear Regression
# 
# A simple linear regression doesn't do better than our dummy regressor! (Alghouth, simple categorical encoding really doesn't make sense for this approach!)

# Simple Linear Regression

# print(f'{score_simple_linear:0.5f}')


# # This seems slow and repetative. Can we automate it a bit?

# def plot_results(name, y, yhat, num_to_plot=10000, lims=(0,12), figsize=(6,6)):
#     plt.figure(figsize=figsize)
#     score = mean_squared_error(y, yhat, squared=False)
#     plt.scatter(y[:num_to_plot], yhat[:num_to_plot])
#     plt.plot(lims, lims)
#     plt.ylim(lims)
#     plt.xlim(lims)
#     plt.title(f'{name}: {score:0.5f}', fontsize=18)
#     plt.show()



# # It look like RandomForest did the best. Let's train it on all the data and make a submission!

model = RandomForestRegressor(n_estimators=50, n_jobs=-1)
# model.fit(train, target)
# submission['target'] = model.predict(test)
# submission.to_csv('random_forest.csv')


# ## Now you should save your Notebook (blue button in the upper right), and then when that's complete go to the notebook viewer and make a submission to the competition. :-)
# 
# ## There's lots of room for improvement. What things can you try to get a better score?
