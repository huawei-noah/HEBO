#!/usr/bin/env python
# coding: utf-8

import numpy as np  # linear algebra
import pandas as pd  # 

# FILE_PATH = "../data/"
FILE_PATH = "./workspace/hyperopt/srhm2/data/"

submission_path = "best_submission.csv"
RANDOM_SEED = 73

macro_economy = pd.read_csv(FILE_PATH + 'macro.csv')
train = pd.read_csv(FILE_PATH + 'train.csv')
test = pd.read_csv(FILE_PATH + 'test.csv')

cat = train.select_dtypes(exclude=['number', 'bool_']).columns
cat_train = train.select_dtypes(exclude=["number", "bool_"]).columns
cat_train = cat_train.drop(['timestamp', 'sub_area', 'ecology'])
train[cat_train] = train[cat_train].apply(lambda x: x.str.replace('yes', '1'))
train[cat_train] = train[cat_train].apply(lambda x: x.str.replace('no', '0'))
train['product_type'] = train['product_type'].apply(lambda x: x.replace('Investment', '1'))
train['product_type'] = train['product_type'].apply(lambda x: x.replace('OwnerOccupier', '0'))

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

encoder = OrdinalEncoder()
train['ecology'] = encoder.fit_transform(train[['ecology']])

train = train.drop(['sub_area'], axis=1)

train[cat_train] = train[cat_train].astype('Float64')
cols = macro_economy.select_dtypes(exclude=["number", "bool_"]).columns
cols = cols[1:]

train['timestamp'] = pd.to_datetime(train['timestamp'])
train['timestamp'] = train['timestamp'].dt.to_period('M')

train = train.dropna(thresh=train.shape[0] * 0.4, axis=1)

train = train.drop(['id'], axis=1)
no_period = train.drop(['timestamp'], axis=1).columns
train[no_period] = train[no_period].fillna(train[no_period].median())

macro_economy = pd.read_csv(FILE_PATH + 'macro_economy.csv')
macro_economy['timestamp'] = pd.to_datetime(macro_economy['timestamp'])
macro_economy['timestamp'] = macro_economy['timestamp'].dt.to_period('M')

final_df = pd.merge(train, macro_economy, how='left', on='timestamp')
final_df = final_df.drop('timestamp', axis=1)


final_df = final_df.loc[:,~final_df.columns.str.contains('5000', case=False)] 
final_df = final_df.loc[:,~final_df.columns.str.contains('3000', case=False)] 
final_df = final_df.loc[:,~final_df.columns.str.contains('2000', case=False)] 

final_df = final_df.drop(columns=['raion_build_count_with_material_info', 'build_count_block',
                        'build_count_wood', 'build_count_frame', 
                        'build_count_brick', 'build_count_monolith',
                        'build_count_panel', 'build_count_foam',
                        'build_count_slag', 'build_count_mix',
                        'ID_railroad_station_walk', 'ID_railroad_station_avto',
                        'ID_big_road1', 'ID_big_road2',
                        'hospital_beds_raion'], axis=1, errors='ignore')

final_df['room_per_sq'] = final_df['life_sq'] / (final_df['num_room'] + 1)
final_df['age'] = 2017 - final_df['build_year']
final_df['floor_per_max'] = final_df['floor'] / (final_df['max_floor'] + 1)
final_df['mortgage_per_income'] = final_df['mortgage_value'] / final_df['income_per_cap']
final_df['pop_density'] = final_df['raion_popul'] / final_df['area_m']

final_df['pop_per_mall'] = final_df['shopping_centers_raion'] / final_df['raion_popul'] 
final_df['pop_per_office'] = final_df['office_raion'] / final_df['raion_popul'] 

final_df['preschool_fill'] = final_df['preschool_quota'] / final_df['children_preschool']
final_df['preschool_capacity'] = final_df['preschool_education_centers_raion'] / final_df['children_preschool']
final_df['school_fill'] = final_df['school_quota'] / final_df['children_school']
final_df['school_capacity'] = final_df['school_education_centers_raion'] / final_df['children_school']

final_df['percent_working'] = final_df['work_all'] / final_df['full_all']
final_df['percent_old'] = final_df['ekder_all'] / final_df['full_all']


features = final_df.drop('price_doc', axis=1).columns
target = final_df[['price_doc']].columns


# ## Feature Scaling

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(final_df[features])


scaled_features = scaler.transform(final_df[features])


scaled_df = pd.DataFrame(scaled_features, columns=features)



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime


X_train, X_val, y_train, y_val = train_test_split(final_df[features], final_df[target], test_size=0.2)


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 10],
        'eta': [0.02, 0.05, 0.1, 0.3],
        'subsample':  [0, 0.2, 0.5],
        'reg_lambda': [0.5, 1, 2],
        'reg_alpha': [0, 0.5, 1]}


xgb = xgb.XGBRegressor(learning_rate=0.02, n_estimators=600, objective='reg:squarederror', eval_metric='rmse',silent=True, nthread=1,device='gpu')



# xgb.fit(X_train, y_train)


# # ## Converting test to format

# # test['product_type'] = test['product_type'].astype('str')


# # cat = test.select_dtypes(exclude=['number', 'bool_']).columns
# # cat_test = test.select_dtypes(exclude=["number","bool_"]).columns
# # cat_test = cat_test.drop(['timestamp', 'sub_area', 'ecology'])
# # test[cat_test] = test[cat_test].apply(lambda x: x.str.replace('yes', '1'))
# # test[cat_test] = test[cat_test].apply(lambda x: x.str.replace('no', '0'))
# # test['product_type'] = test['product_type'].apply(lambda x: x.replace('Investment', '1'))
# # test['product_type'] = test['product_type'].apply(lambda x: x.replace('OwnerOccupier', '0'))


# # test['ecology'] = encoder.transform(test[['ecology']])


# # test = test.drop(['sub_area'], axis=1)

# # test[cat_train] = test[cat_test].astype('Float64')
# # test['timestamp'] = pd.to_datetime(test['timestamp'])
# # test['timestamp'] = test['timestamp'].dt.to_period('M')
# # no_period = test.drop(['timestamp', 'id'], axis=1).columns
# # test[no_period] = test[no_period].fillna(test[no_period].median())
# # test_df = pd.merge(left=test, right=macro_economy, how='left', on='timestamp')
# # id = test_df['id']
# # test_df = test_df.drop(['timestamp', 'id'], axis=1)
# # test_df = test_df.astype('Float64')
# # a = test_df.values.astype('float64')


# # test_df = test_df.loc[:,~test_df.columns.str.contains('5000', case=False)] 
# # test_df = test_df.loc[:,~test_df.columns.str.contains('3000', case=False)] 
# # test_df = test_df.loc[:,~test_df.columns.str.contains('2000', case=False)] 

# # test_df = test_df.drop(columns=['raion_build_count_with_material_info', 'build_count_block',
# #                         'build_count_wood', 'build_count_frame', 
# #                         'build_count_brick', 'build_count_monolith',
# #                         'build_count_panel', 'build_count_foam',
# #                         'build_count_slag', 'build_count_mix',
# #                         'ID_railroad_station_walk', 'ID_railroad_station_avto',
# #                         'ID_big_road1', 'ID_big_road2',
# #                         'hospital_beds_raion'], axis=1, errors='ignore')

# # test_df['room_per_sq'] = test_df['life_sq'] / (test_df['num_room'] + 1)
# # test_df['age'] = 2017 - test_df['build_year']
# # test_df['floor_per_max'] = test_df['floor'] / (test_df['max_floor'] + 1)
# # test_df['mortgage_per_income'] = test_df['mortgage_value'] / test_df['income_per_cap']
# # test_df['pop_density'] = test_df['raion_popul'] / test_df['area_m']

# # test_df['pop_per_mall'] = test_df['shopping_centers_raion'] / test_df['raion_popul'] 
# # test_df['pop_per_office'] = test_df['office_raion'] / test_df['raion_popul'] 

# # test_df['preschool_fill'] = test_df['preschool_quota'] / test_df['children_preschool']
# # test_df['preschool_capacity'] = test_df['preschool_education_centers_raion'] / test_df['children_preschool']
# # test_df['school_fill'] = test_df['school_quota'] / test_df['children_school']
# # test_df['school_capacity'] = test_df['school_education_centers_raion'] / test_df['children_school']

# # test_df['percent_working'] = test_df['work_all'] / test_df['full_all']
# # test_df['percent_old'] = test_df['ekder_all'] / test_df['full_all']


# # predictions = pd.DataFrame(xgb.predict(test_df))


# # predictions['id'] = pd.Series(id)


# # predictions = predictions.rename(columns={0: 'price_doc'})
# # predictions


# # predictions = predictions[['id', 'price_doc']]
# # predictions.to_csv('submission.csv', index=False)






# score=xgb.score(X_val, y_val)
# print(score)

