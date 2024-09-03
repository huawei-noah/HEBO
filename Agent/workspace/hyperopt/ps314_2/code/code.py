#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# sns.set_theme(style = 'white', palette = 'viridis')
# pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)

FILE_PATH = "./workspace/hyperopt/ps314_2/data/"    

train = pd.read_csv(FILE_PATH+'train.csv.zip')
test_1 = pd.read_csv(FILE_PATH+'test.csv.zip')
# orig_train = pd.read_csv(r'../input/wild-blueberry-yield-prediction-dataset/WildBlueberryPollinationSimulationData.csv')

train.drop('id', axis = 1, inplace = True)
test = test_1.drop('id', axis = 1)
# orig_train.drop('Row#', axis = 1, inplace = True)


# # Knowing Your Data
# 
# ## Descriptive Statistics

# train.head(10)


# desc = train.describe().T
# desc['nunique'] = train.nunique()
# desc['%unique'] = desc['nunique'] / len(train) * 100
# desc['null'] = train.isna().sum()
# desc['type'] = train.dtypes
# desc


# desc = test.describe().T
# desc['nunique'] = test.nunique()
# desc['%unique'] = desc['nunique'] / len(train) * 100
# desc['null'] = test.isna().sum()
# desc['type'] = test.dtypes
# desc


# desc = orig_train.describe().T
# desc['nunique'] = orig_train.nunique()
# desc['%unique'] = desc['nunique'] / len(orig_train) * 100
# desc['null'] = orig_train.isna().sum()
# desc['type'] = orig_train.dtypes
# desc


# # Duplicates

# print(f'There are {train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {train.count()[0]} rows in train dataset')
# print(f'There are {test.duplicated().value_counts()[0]} non-duplicate values out of {test.count()[0]} rows in test dataset')
# print(f'There are {orig_train.duplicated(subset = list(train)[0:-1]).value_counts()[0]} non-duplicate values out of {orig_train.count()[0]} rows in original train dataset')


# # **Key point**: There are row duplicates in train and test dataset. We can remove it from our train dataset, though it may have no effect due to how few they are.

# # # Adversarial Validation

# def adversarial_validation(dataset_1 = train, dataset_2 = test, label = 'Train-Test'):

#     adv_train = dataset_1.drop('yield', axis = 1)
#     adv_test = dataset_2.copy()

#     adv_train['is_test'] = 0
#     adv_test['is_test'] = 1

#     adv = pd.concat([adv_train, adv_test], ignore_index = True)

#     adv_shuffled = adv.sample(frac = 1)

#     adv_X = adv_shuffled.drop('is_test', axis = 1)
#     adv_y = adv_shuffled.is_test

#     skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

#     val_scores = []
#     predictions = np.zeros(len(adv))

#     for fold, (train_idx, val_idx) in enumerate(skf.split(adv_X, adv_y)):
    
#         adv_lr = XGBClassifier(random_state = 42)    
#         adv_lr.fit(adv_X.iloc[train_idx], adv_y.iloc[train_idx])
        
#         val_preds = adv_lr.predict_proba(adv_X.iloc[val_idx])[:,1]
#         predictions[val_idx] = val_preds
#         val_score = roc_auc_score(adv_y.iloc[val_idx], val_preds)
#         val_scores.append(val_score)
    
#     fpr, tpr, _ = roc_curve(adv['is_test'], predictions)
    
#     plt.figure(figsize = (10, 10), dpi = 300)
#     sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Indistinguishable Datasets")
#     sns.lineplot(x=fpr, y=tpr, label="Adversarial Validation Classifier")
#     plt.title(f'{label} Validation = {np.mean(val_scores):.5f}', weight = 'bold', size = 17)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.show()


# adversarial_validation()
# adversarial_validation(pd.concat([train, orig_train]), test, 'Combo Train-Test Validation')


# # **Key points:**
# # 1. Train and test datasets validation results in ROC score of close to .5, therefore **we can trust our cross-validation.**
# # 2. Combined train and test datasets validation results in ROC score of close to .5, which is very far from competition dataset. Therefore, **we can include it in our training**.

# # # Distribution

# fig, ax = plt.subplots(4, 4, figsize = (10, 10), dpi = 300)
# ax = ax.flatten()

# for i, column in enumerate(test.columns):
#     sns.kdeplot(train[column], ax=ax[i], color=pal[0])    
#     sns.kdeplot(test[column], ax=ax[i], color=pal[2])
    
#     ax[i].set_title(f'{column} Distribution', size = 7)
#     ax[i].set_xlabel(None)
    
# fig.suptitle('Distribution of Feature\nper Dataset\n', fontsize = 24, fontweight = 'bold')
# fig.legend(['Train', 'Test'])
# plt.tight_layout()


# # **Key points:** 
# # 1. All features have similar distribution between training and test dataset.
# # 2. 13 out of 16 features are categorical

# plt.figure(figsize = (10, 6), dpi = 300)
# sns.kdeplot(data = train, x = 'yield')
# plt.title('Target Distribution', weight = 'bold', size = 20)
# plt.show()


# # **Key point**: It looks like we are having relatively normal distribution here.

# # # Correlation

# def heatmap(dataset, label = None):
#     corr = dataset.corr(method = 'spearman')
#     plt.figure(figsize = (14, 10), dpi = 300)
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     sns.heatmap(corr, mask = mask, cmap = 'viridis', annot = True, annot_kws = {'size' : 7})
#     plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
#     plt.show()


# heatmap(train, 'Train')
# heatmap(test, 'Test')


# # **Key point**: There are so many features with very strong correlation that some of them are practically duplicates. We can remove them to make our model better. Let's try to see it with hierarchy tree this time.

# def distance(data, label = ''):
#     #thanks to @sergiosaharovsky for the fix
#     corr = data.corr(method = 'spearman')
#     dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
    
#     plt.figure(figsize = (10, 8), dpi = 300)
#     dendro = dendrogram(dist_linkage, labels=data.columns, leaf_rotation=90)
#     plt.title(f'Feature Distance in {label} Dataset', weight = 'bold', size = 22)
#     plt.show()


# distance(train, 'Train')


# **Key points:** 
# 1. `MinOfUpperTRange`, `AverageOfUpperTRange`, `AverageOfLowerTRange`, `MaxOfLowerTRange`, `MaxOfUpperTRange`, and `MinOfLowerTRange` are practically duplicates so you can just keep one of them.
# 2. `RainingDays` and `AverageRainingDays` are almost duplicate so you may also drop one of them.

# # Preparation

X = train.copy()
y = X.pop('yield')

seed = 42
# splits = 5
# k = KFold(n_splits = splits, random_state = seed, shuffle = True)

np.random.seed(seed)


# # # Base Models

# def cross_val_score(model, cv = k, label = ''):
    
#     X = train.copy()
#     y = X.pop('yield')
    
#     #initiate prediction arrays and score lists
#     val_predictions = np.zeros((len(train)))
#     train_predictions = np.zeros((len(train)))
#     train_mae, val_mae = [], []
    
#     #training model, predicting prognosis probability, and evaluating log loss
#     for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])

#         train_preds = model.predict(X.iloc[train_idx])
#         val_preds = model.predict(X.iloc[val_idx])
                  
#         train_predictions[train_idx] += train_preds
#         val_predictions[val_idx] += val_preds
        
#         train_score = mean_absolute_error(y.iloc[train_idx], train_preds)
#         val_score = mean_absolute_error(y.iloc[val_idx], val_preds)
        
#         train_mae.append(train_score)
#         val_mae.append(val_score)
    
#     print(f'Val MAE: {np.mean(val_mae):.5f} ± {np.std(val_mae):.5f} | Train MAE: {np.mean(train_mae):.5f} ± {np.std(train_mae):.5f} | {label}')
    
#     return val_mae


# mae_list = pd.DataFrame()

# models = [
#     ('linear', LinearRegression()),
#     ('ridge', Ridge(random_state = seed)),
#     ('lasso', Lasso(random_state = seed, max_iter = 1000000)),
#     ('elastic', ElasticNet(random_state = seed, max_iter = 1000000)),
#     ('huber', HuberRegressor(max_iter = 1000000)),
#     ('ard', ARDRegression()),
#     ('passive', PassiveAggressiveRegressor(random_state = seed)),
#     ('theilsen', TheilSenRegressor(random_state = seed)),
#     ('linearsvm', LinearSVR(random_state = seed, max_iter = 1000000)),
#     ('mlp', MLPRegressor(random_state = seed, max_iter = 1000000)),
#     ('et', ExtraTreesRegressor(random_state = seed)),
#     ('rf', RandomForestRegressor(random_state = seed)),
#     ('xgb', XGBRegressor(random_state = seed, eval_metric = 'mae')),
#     ('lgb', LGBMRegressor(random_state = seed, objective = 'mae')),
#     ('dart', LGBMRegressor(random_state = seed, boosting_type = 'dart')),
#     ('cb', CatBoostRegressor(random_state = seed, objective = 'MAE', verbose = 0)),
#     ('gb', GradientBoostingRegressor(random_state = seed, loss = 'absolute_error')),
#     ('hgb', HistGradientBoostingRegressor(random_state = seed, loss = 'absolute_error')),
#     ('knn', KNeighborsRegressor())
# ]

# for (label, model) in models:
#      mae_list[label] = cross_val_score(model, label = label)


# plt.figure(figsize = (8, 4), dpi = 300)
# sns.barplot(data = mae_list.reindex((mae_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
# plt.title('MAE Comparison', weight = 'bold', size = 20)
# plt.show()


# # **Key points**;
# # 1. Linear regression can work as well as tree-based models here.
# # 2. Some tree-based models, especially non-gradient boosting ones, have a lot of overfitting.
# # 3. `CatBoostRegressor` gives the best result.

# # # Base Models 2.0 (With Post-Processing)
# # 
# # @mattop has pointed out in [this topic](https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327) that we can post-process our prediction to make it consistent with the unique values of the `yield`.

# def postprocessor(prediction):
#     #thanks to @mattop
#     unique_targets = np.unique(train['yield'])
#     return [min(unique_targets, key = lambda x: abs(x - pred)) for pred in prediction]


# def cross_val_score_2(model, cv = k, label = ''):
    
#     X = train.copy()
#     y = X.pop('yield')
    
#     #initiate prediction arrays and score lists
#     val_predictions = np.zeros((len(train)))
#     train_predictions = np.zeros((len(train)))
#     train_mae, val_mae = [], []
    
#     #training model, predicting prognosis probability, and evaluating log loss
#     for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])

#         train_preds = postprocessor(model.predict(X.iloc[train_idx]))
#         val_preds = postprocessor(model.predict(X.iloc[val_idx]))
                  
#         train_predictions[train_idx] += train_preds
#         val_predictions[val_idx] += val_preds
        
#         train_score = mean_absolute_error(y.iloc[train_idx], train_preds)
#         val_score = mean_absolute_error(y.iloc[val_idx], val_preds)
        
#         train_mae.append(train_score)
#         val_mae.append(val_score)
    
#     print(f'Val MAE: {np.mean(val_mae):.5f} ± {np.std(val_mae):.5f} | Train MAE: {np.mean(train_mae):.5f} ± {np.std(train_mae):.5f} | {label}')
    
#     return val_mae


# for (label, model) in models:
#     mae_list[label] = cross_val_score_2(
#         model,
#         label = label
#     )


# plt.figure(figsize = (8, 4), dpi = 300)
# sns.barplot(data = mae_list.reindex((mae_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
# plt.title('MAE Comparison', weight = 'bold', size = 20)
# plt.show()


# # **Key point:** It seems that we have a miniscule, but consistent improvement across our models on the MAE score.

# # # Base Model 3.0 (Postprocessing + Scaling)

# for (label, model) in models:
#     mae_list[label] = cross_val_score_2(
#         Pipeline([
#             ('scale', StandardScaler()),
#             (label, model)]),
#         label = label
#     )


# plt.figure(figsize = (8, 4), dpi = 300)
# sns.barplot(data = mae_list.reindex((mae_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
# plt.title('MAE Comparison', weight = 'bold', size = 20)
# plt.show()


# **Key point**: There is consistent improvement after we scale the features, especially on non-tree-based models.

# # Ensemble
# 
# Now let's try to build a simple average ensemble. For simplicity, we will only use LightGBM and CatBoost, which are the best 2 models here.
from sklearn.metrics import mean_absolute_error

ensemble_models = [
    ('lgb', LGBMRegressor(random_state = seed, objective = 'mae')),
    ('cb', CatBoostRegressor(random_state = seed, objective = 'MAE', verbose = 0))
]

voter = Pipeline([('scale', StandardScaler()), ('vote',VotingRegressor(ensemble_models))])

# _ = cross_val_score_2(voter, label = 'Voting Ensemble')


# # **Key point**: It looks like our score has improved with the ensemble, from **343.11222** as the best score from our baseline CatBoost, to **341.60802** from simple average ensemble with scaling and post-processing.

# # # Modeling

# voter.fit(X, y)
# prediction = postprocessor(voter.predict(test))


# # # Submission

# test_1.drop(list(test_1.drop('id', axis = 1)), axis = 1, inplace = True)


# test_1['yield'] = prediction
# test_1.to_csv('submission.csv', index = False)


# Thank you for reading!
