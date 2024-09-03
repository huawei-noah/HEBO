#!/usr/bin/env python
# coding: utf-8

# # 📂 Imports 📂

# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import optuna
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor, plot_importance
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from imblearn.pipeline import Pipeline

from itertools import combinations


# # 📈 Exploratory Data Analysis 📊

# ### Importing Data + a High-Level Look at the Data

# import the data
FILE_PATH = "./workspace/hyperopt/ps315_2/data/"    

all_data = pd.read_csv(FILE_PATH+'data.csv.zip')

# separate data into train and submission sets based on blank target values
train = all_data[all_data['x_e_out [-]'].isna() == False]
submission = all_data[all_data['x_e_out [-]'].isna() == True]

# # get length of train and test datasets
# print(f'\nTrain dataset length: {train.shape[0]}')
# print(f'Submission dataset length: {submission.shape[0]}\n')

# # check for missing values
# print(f'There are {int(train.isna().sum().sum())} missing feature values in the train set.')
# print(f'There are {int(submission.isna().sum().sum())} missing feature values in the submission set.\n')

# # check for duplicate rows
# n_duplicate_rows = len(train) - len(train.drop_duplicates())
# print(f'There are {int(n_duplicate_rows)} duplicate rows in the train dataset.\n')

# # quick high-level overview of dataset
# pd.set_option('display.expand_frame_repr', False) # need this because there are so many features
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# display(train.head())
# print('\n\n')
# display(train.describe().round(decimals=2))


# #### 💡Insights: First Glance
# - Train dataset is approximately twice as large as the submission dataset
# - The train dataset is missing a significant amount of data, averaging over one missing feature value per data point
# - The submission dataset is missing nearly twice as much data, averaging nearly two missing feature values per data point
# - There are no duplicate rows in the training dataset
# - D_h and chf_exp seem to have some significant outliers on the upper end

# ### Renaming Featurse + Creating lists of columns by feature type

# renaming columns to something more succinct and readable
column_renaming_dict = {'pressure [MPa]': 'pressure',
                        'mass_flux [kg/m2-s]': 'mass_flux',
                        'x_e_out [-]': 'x_e_out',
                        'D_e [mm]': 'D_e',
                        'D_h [mm]': 'D_h',
                        'length [mm]': 'length',
                        'chf_exp [MW/m2]': 'chf_exp'}

train = train.rename(columns=column_renaming_dict)
submission = submission.rename(columns=column_renaming_dict)
# display(train.head())
                        
# creating groups by feature type
features = {'continuous': ['pressure', 'mass_flux', 'D_e', 'D_h', 'length', 'chf_exp'],
            'categorical': ['author', 'geometry']}


# ### Checking target distribution

# fix, ax = plt.subplots(figsize=(6, 6))
# sns.kdeplot(data=train, x='x_e_out', fill=True, ax=ax).set_title('Target Distribution on Train Set');
# ax = np.ravel(ax)
# ax[0].grid(visible=True)


# creating a log transformation of the target
transformed_target = np.power(10, train[['x_e_out']]) - 1

# plotting distribution
# fix, ax = plt.subplots(figsize=(6, 6))
# sns.kdeplot(data=transformed_target, x='x_e_out', fill=True, ax=ax).set_title('Transformed Target Distribution on Train Set [10^x - 1]');
# ax = np.ravel(ax)
# ax[0].grid(visible=True)

# adding this to the dataframe
train['log_x_e_out'] = transformed_target


# #### 💡Insights: Target Distribution
# - The target distribution on the training data has some left skewness, but it does not seem too severe
# - Transforming the target with 10^x seems to make the distribution much more Gaussian. It may be worth looking into this to see if it helps the predictions

# ### Checking Feature Distribution

# plotting distribution of each continuous feature in train and test datasets
# fig, ax = plt.subplots(2, 3, figsize=(20, 10))
# ax = np.ravel(ax)
# palette = sns.color_palette('coolwarm', 2)

# for i, col in enumerate(features['continuous']):
#     sns.kdeplot(data=train, x=train[col], ax=ax[i], label='Train', color=palette[0], fill=True)
#     sns.kdeplot(data=submission, x=submission[col], ax=ax[i], label='Test', color=palette[1], fill=True)
#     ax[i].set_title(f'{col}', fontsize=12)
#     ax[i].legend(title='Dataset', loc='upper right', labels=['Train', 'Test'])
    
# fig.suptitle('Continuous Feature Distributions (Train & Test)', fontsize=20);
# fig.tight_layout(pad=3)


# creating function to create a distribution histogram for each discrete value
# def create_dist_barplot(train_df, test_df, feature_name, ax):
#     train_value_counts = pd.DataFrame(train_df.value_counts(feature_name, normalize=True))
#     train_value_counts['Distribution'] = ['Train'] * train_value_counts.shape[0]
#     test_value_counts = pd.DataFrame(test_df.value_counts(feature_name, normalize=True))
#     test_value_counts['Distribution'] = ['Test'] * test_value_counts.shape[0]
#     barplot_df = pd.concat([train_value_counts, test_value_counts], axis=0)
#     barplot_df = barplot_df.rename(columns={'proportion': 'Density'})
#     barplot_df = barplot_df.reset_index()
#     sns.barplot(data=barplot_df, x=feature_name, y='Density', hue='Distribution', ax=ax, palette='coolwarm')

# # plotting distribution of each integer feature in train and test datasets
# fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# ax = np.ravel(ax)
# palette = sns.color_palette('coolwarm', 2)

# for i, col in enumerate(features['categorical']):
#     create_dist_barplot(train, submission, col, ax[i])
#     ax[i].set_title(f'{col}', fontsize=14)
    
# fig.suptitle('Categorical Feature Distributions (Train & Test)', fontsize=20);
# fig.tight_layout(pad=1)


# #### 💡Insights: Feature Distributions:
# - The feature distributions between the train and test datasets seem to be extremely similar for both categorical and numerical features

# adjusting distributions of skewed features
train['D_e'] = np.log1p(train['D_e'])
train['D_h'] = np.log1p(train['D_h'])
train['length'] = np.log1p(train['length'])
train['chf_exp'] = np.log1p(train['chf_exp'])

submission['D_e'] = np.log1p(submission['D_e'])
submission['D_h'] = np.log1p(submission['D_h'])
submission['length'] = np.log1p(submission['length'])
submission['chf_exp'] = np.log1p(submission['chf_exp'])

# plotting distribution of each continuous feature in train and test datasets
# fig, ax = plt.subplots(2, 3, figsize=(20, 10))
# ax = np.ravel(ax)
# palette = sns.color_palette('coolwarm', 2)

# for i, col in enumerate(features['continuous']):
#     sns.kdeplot(data=train, x=train[col], ax=ax[i], label='Train', color=palette[0], fill=True)
#     sns.kdeplot(data=submission, x=submission[col], ax=ax[i], label='Test', color=palette[1], fill=True)
#     ax[i].set_title(f'{col}', fontsize=12)
#     ax[i].legend(title='Dataset', loc='upper right', labels=['Train', 'Test'])
    
# fig.suptitle('Continuous Feature Distributions (Train & Test)', fontsize=20);
# fig.tight_layout(pad=3)


# ### Examining Feature Correlation

# calculating the raw correlation matrix
raw_correlation = train[features['continuous'] + ['x_e_out']].corr()

# only keeping the lower diagonal
correlation = raw_correlation.copy()
mask = np.zeros_like(correlation, dtype=bool)
mask[np.triu_indices_from(mask)] = True
correlation[mask] = np.nan

# plotting
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', xticklabels=True, yticklabels=True, ax=ax, vmin=-1, vmax=1).set_title('Correlation Matrix', fontsize=20);


# showing pairplot for continuous features
pairplot = sns.pairplot(data=train, vars=features['continuous'], diag_kind='kde');
pairplot.fig.suptitle('Pairplot for Continuous Features on Train Data', y=1.03, fontsize=20);


# #### 💡Insights: Feature Correlation
# - After transforming some of the features, *D_e* and *D_h* are highly correlated. Before transformation, there weren't any highly correlated features
# - *D_h* is somewhat negatively correlated to pressure and positively correlated to *D_e*
# - *D_e* is also somewhat negatively correlated to pressure, which makes sense considering it is positively correlated with *D_h*
# - Looking at the pairplots, there is a strange correlation between *D_e* and *D_h*. They appear to have a perfect linear correlation with some random noise sprinkled in...

# # 📏 Feature Engineering 📐

# ### Imputing missing categorical feature values and testing encoding technique

# add missing labels using the mode for each the categories
train_imputed = train.copy(deep=True)
train_imputed['author'] = train_imputed['author'].replace(np.nan, 'Thompson')
train_imputed['geometry'] = train_imputed['geometry'].replace(np.nan, 'tube')

submission_imputed = submission.copy(deep=True)
submission_imputed['author'] = submission_imputed['author'].replace(np.nan, 'Thompson')
submission_imputed['geometry'] = submission_imputed['geometry'].replace(np.nan, 'tube')

# show the difference
# display(train.head())
# display(train_imputed.head())

# create a categorical one-hot encoder
categorical_transformer = ColumnTransformer(transformers=[('onehot1', OneHotEncoder(sparse_output=False), ['author']),
                                                          ('onehot2', OneHotEncoder(sparse_output=False), ['geometry']),
                                                          ('passthrough', 'passthrough', features['continuous'])],
                                            verbose_feature_names_out=False)
categorical_transformer.set_output(transform='pandas')

# pass the data through the encoder
train_cat_onehot_test = categorical_transformer.fit_transform(train_imputed)
# display(train_cat_onehot_test.head())


# ### Creating Linear Regression based Imputer for D_e and D_h since they are highly correlated

# creating a custom transformer
class D_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.D_e_regressor = LinearRegression()
        self.D_h_regressor = LinearRegression()
        
    def _impute_D_e(self, row):
        D_e = row['D_e']
        D_h = row['D_h']
        if np.isnan(D_e) and np.isnan(D_h)==False:
            D_e = self.D_e_regressor.predict(np.reshape(np.array(D_h), (-1, 1)))
        return float(D_e)

    def _impute_D_h(self, row):
        D_e = row['D_e']
        D_h = row['D_h']
        if np.isnan(D_h) and np.isnan(D_e)==False:
            D_h = self.D_h_regressor.predict(np.reshape(np.array(D_e), (-1, 1)))
        return float(D_h)
    
    def fit(self, X, y=None):
        # gathering D_e and D_h data where both are not NaN
        complete_D_data = X[['D_h', 'D_e']]
        filtered_D_data = complete_D_data[complete_D_data.isna().T.any() == False]

        D_e_array = np.reshape(np.array(filtered_D_data['D_e']), (-1, 1))
        D_h_array = np.reshape(np.array(filtered_D_data['D_h']), (-1, 1))

        # fitting regressors for each based on complete data
        self.D_e_regressor.fit(D_h_array, D_e_array)
        self.D_h_regressor.fit(D_e_array, D_h_array)
        
        return self
        
    def transform(self, X, y=None):
        X['D_e'] = X.apply(lambda row: self._impute_D_e(row), axis=1)
        X['D_h'] = X.apply(lambda row: self._impute_D_h(row), axis=1)
        
        return X


# ### Splitting "train" data into Train and Test Sets

target_name = 'x_e_out'
features_to_include = features['continuous'] + features['categorical']
X_train = train_imputed[features_to_include]
y_train = train_imputed[target_name]

# splitting training data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(train_imputed[features_to_include],
#                                                     train_imputed[target_name],
#                                                     train_size=0.80,
#                                                     shuffle=True,
#                                                     random_state=1)

# print(f'Size of X_train: {X_train.shape}\nSize of y_train: {y_train.shape}')
# print(f'Size of X_test: {X_test.shape}\nSize of y_test: {y_test.shape}')


# ### Building a baseline model and assessing features

# start with XGBoost
X_baseline = X_train
y_baseline = y_train

# create a categorical one-hot encoder
categorical_transformer = ColumnTransformer(transformers=[('onehot1', OneHotEncoder(sparse_output=False), ['author']),
                                                          ('onehot2', OneHotEncoder(sparse_output=False), ['geometry']),
                                                          ('passthrough', 'passthrough', features['continuous'])],
                                            verbose_feature_names_out=False)
categorical_transformer.set_output(transform='pandas')

# create a regression imputer for D_h and D_e
D_imputer = D_transformer()

# create an imputer
# simple_imputer = SimpleImputer(strategy='median', copy=True)
# simple_imputer.set_output(transform='pandas')

# # create a baseline model to compare with
# baseline_model = XGBRegressor(gamma=0.04, reg_lambda=0.04) # quickly add in some regularization by trial and error to prevent extreme overfitting

# # create the pipeline
# baseline_pipeline = Pipeline([('cat_transformer', categorical_transformer),
#                               ('D_imputer', D_imputer),
#                               ('imputer', simple_imputer),
#                               ('regressor', baseline_model)])
# cv_results = cross_validate(baseline_pipeline, X_baseline, y_baseline, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
# mean_test_score = np.mean(cv_results['test_score'])
# train_test_score_rmse = np.std(cv_results['test_score']) # helpful to measure variance

# # print scoring metrics
# print('Test Scores on K-Folds: ' + str(np.round(cv_results['test_score'], decimals=3)))
# print('Train Scores on K-Folds: ' + str(np.round(cv_results['train_score'], decimals=3)))
# print(f'Mean Test Score: {np.round(mean_test_score, decimals=5)}')
# print(f'Test Score K-Fold Std: {np.round(train_test_score_rmse, decimals=5)}')


# add a feature of random noise to help judge feature importances
X_baseline['random_noise'] = np.random.normal(size=X_baseline.shape[0])

# create a categorical one-hot encoder that includes random noise
categorical_transformer_noise = ColumnTransformer(transformers=[('onehot1', OneHotEncoder(sparse_output=False), ['author']),
                                                                ('onehot2', OneHotEncoder(sparse_output=False), ['geometry']),
                                                                ('passthrough', 'passthrough', features['continuous'] + ['random_noise'])],
                                            verbose_feature_names_out=False)
categorical_transformer_noise.set_output(transform='pandas')



# #### 💡Insights: Baseline Model
# - Imputing missing categorical features with the mode seemed to work well
# - Imputing missing continuous features with the median seemed to work well also
# - The baseline XGBoost score seemed to perform fairly well and had much less variance than expected
# - The most important feature by far seems to be *chf_exp*, which out of all features was highest correlated to the target. *pressure*, *D_e*, *D_h*, *length*, and some *author* one-hots follow.
# - A random noise feature was added to use as a reference for how useful features are. All of the continuous features were above this threshold, but some *author* one-hots and all of the *geometry* one-hots were below the threshold.
# - *geometry* may be a good candidate as a feature to drop
# - The ICE plots show some odd behavior with the *D_e* and *D_h* features and their effect on the target.
# - *chf_exp* and *pressure* seem to have a visible negative correlation with the target. *length* seems to have a less apparent positive correlation. This matches the Pearson coefficients calculated earlier

# ### Testing alternative imputation techniques

# create a categorical one-hot encoder
categorical_transformer = ColumnTransformer(transformers=[('onehot1', OneHotEncoder(sparse_output=False), ['author']),
                                                          ('onehot2', OneHotEncoder(sparse_output=False), ['geometry']),
                                                          ('passthrough', 'passthrough', features['continuous'])],
                                            verbose_feature_names_out=False)
categorical_transformer.set_output(transform='pandas')

# create an imputer
knn_imputer = KNNImputer(n_neighbors=3, weights='uniform', copy=True)
knn_imputer.set_output(transform='pandas')
# iterative_imputer = IterativeImputer(max_iter=10, imputation_order='descending')

# create an imputer
imputer = SimpleImputer(strategy='median', copy=True)
imputer.set_output(transform='pandas')



# create the XGBoost object
xgb_final_1 = XGBRegressor(n_estimators=267,
                           max_depth=7,
                           min_child_weight=5.93313,
                           gamma=0.002317,
                           learning_rate=0.034267,
                           subsample=0.60232,
                           colsample_bytree=0.62122,
                           reg_lambda=1.39154)

# # create the pipeline


# create the XGBoost object
xgb_final_2 = XGBRegressor(n_estimators=245,
                           max_depth=7,
                           min_child_weight=5.88154,
                           gamma=0.0024124,
                           learning_rate=0.035098,
                           subsample=0.62636,
                           colsample_bytree=0.61926,
                           reg_lambda=1.30091)

# # create the pipeline
# xgb_pipeline_2 = Pipeline([('cat_transformer', categorical_transformer),
#                            ('D_imputer', D_imputer),
#                            ('imputer', imputer),
#                            ('regressor', xgb_final_2)])
# 

# create the LightGBM object
lgbm_final_1 = LGBMRegressor(n_estimators=176,
                             learning_rate=0.031217,
                             num_leaves=2681,
                             max_depth=11,
                             min_child_weight=0.03876,
                             min_child_samples=47,
                             subsample=0.61635,
                             colsample_bytree=0.510339,
                             reg_lambda=0.0082346)

# # create the pipeline

# create the LightGBM object
lgbm_final_2 = LGBMRegressor(n_estimators=223,
                             learning_rate=0.029477,
                             num_leaves=2618,
                             max_depth=10,
                             min_child_weight=0.028960,
                             min_child_samples=49,
                             subsample=0.63466,
                             colsample_bytree=0.52098,
                             reg_lambda=0.007196)

# # create the pipeline

# ### Creating an ensemble model
from sklearn.metrics import mean_squared_error
# creating a voting ensemble from the models
voting_model = VotingRegressor(estimators=[('xgb_1', xgb_final_1), ('xgb_2', xgb_final_2), ('lgbm_1', lgbm_final_1), ('lgbm_2', lgbm_final_2)])
voting_pipeline = Pipeline([('cat_transformer', categorical_transformer),
                            ('D_imputer', D_imputer),
                            ('imputer', imputer),
                            ('regressor', voting_model)])

# # begin cross-validation
# cv_results = cross_validate(voting_pipeline, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
# mean_test_score = np.mean(cv_results['test_score'])
# train_test_score_rmse = np.std(cv_results['test_score']) # helpful to measure variance

# # print scoring metrics
# print('Test Scores on K-Folds: ' + str(np.round(cv_results['test_score'], decimals=3)))
# print('Train Scores on K-Folds: ' + str(np.round(cv_results['train_score'], decimals=3)))
# print(f'Mean Test Score: {np.round(mean_test_score, decimals=5)}')
# print(f'Test Score K-Fold Std: {np.round(train_test_score_rmse, decimals=5)}')

# # fit the pipeline to the training data and report score on test data
# voting_pipeline.fit(X_train, y_train)
# y_test_pred = voting_pipeline.predict(X_test)
# model_score = mean_squared_error(y_test, y_test_pred, squared=False)
# print(f'Test Data Score: {np.round(model_score, decimals=5)}')


# # creating a stacked ensemble from the models
# stacked_model = StackingRegressor(estimators=[('xgb_1', xgb_final_1), ('xgb_2', xgb_final_2), ('lgbm_1', lgbm_final_1), ('lgbm_2', lgbm_final_2)], final_estimator=BayesianRidge())
# stacked_pipeline = Pipeline([('cat_transformer', categorical_transformer),
#                              ('D_imputer', D_imputer),
#                              ('imputer', imputer),
#                              ('regressor', stacked_model)])

# begin cross-validation
# cv_results = cross_validate(stacked_pipeline, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error', return_train_score=True)
# mean_test_score = np.mean(cv_results['test_score'])
# train_test_score_rmse = np.std(cv_results['test_score']) # helpful to measure variance

# # print scoring metrics
# print('Test Scores on K-Folds: ' + str(np.round(cv_results['test_score'], decimals=3)))
# print('Train Scores on K-Folds: ' + str(np.round(cv_results['train_score'], decimals=3)))
# print(f'Mean Test Score: {np.round(mean_test_score, decimals=5)}')
# print(f'Test Score K-Fold Std: {np.round(train_test_score_rmse, decimals=5)}')

# fit the pipeline to the training data and report score on test data
# stacked_pipeline.fit(X_train, y_train)
# y_test_pred = stacked_pipeline.predict(X_test)
# model_score = mean_squared_error(y_test, y_test_pred, squared=False)
# print(f'Test Data Score: {np.round(model_score, decimals=5)}')


# # # 📦 Submission 📦

# # ### Make Predictions

# # make predictions
# X_submission = submission_imputed[features['continuous'] + features['categorical']]
# y_submission_pred = voting_pipeline.predict(X_submission)

# # formatting predictions for submission file output
# submission_df = pd.DataFrame({'id': submission_imputed['id'], 'x_e_out [-]': y_submission_pred})
# display(submission_df.head())

# # saving predictions to .csv file for submission
# submission_df.to_csv('submission.csv', header=True, index=False)




