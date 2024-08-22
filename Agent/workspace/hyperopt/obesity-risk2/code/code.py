#!/usr/bin/env python
# coding: utf-8

import os
import random as rn
os.environ['PYTHONHASHSEED'] = '51'
rn.seed(89)

# # Introduction
# <div style="font-size:120%"> 
#     <b>Goal:</b> We have to predict obesity risk in individuals.<br><br>
#     <b>Dataset Description:</b>
# </div>
# 
# | Column | Full Form | Description| 
# |---|---|---|
# | 'id'| id | Unique for each person(row)|
# |'Gender'| Gender| person's Gender|
# | 'Age' | Age| Dtype is float. Age is between 14 years to 61 years |
# |'Height'| Height | Height is in meter it's between 1.45m to 1.98m|
# | 'Weight' | Weight| Weight is between 39 to 165. I think it's in KG.|
# |'family_history_with_overweight'| family history <br> with overweight| yes or no question|
# | 'FAVC'| Frequent consumption <br> of high calorie food| it's yes or no question. i think question they asked is <br>do you consume high calorie food|
# |'FCVC'|  Frequency of <br>consumption of vegetables| Similar to FAVC. this is also `yes or no` question|
# |'NCP'| Number of main meals| dtype is float, NCP is between 1 & 4. I think it should be 1,2,3,4 <br>but our data is synthetic so it's taking float values|
# |'CAEC'| Consumption of <br>food between meals| takes 4 values `Sometimes`, `Frequently`, `no` & `Always` <br>|
# | 'SMOKE'| Smoke | yes or no question. i think the question is "Do you smoke?" |
# |'CH2O'| Consumption of <br>water daily| CH2O takes values between 1 & 3. again it's given as <br>float may be because of synthetic data. it's values should be 1,2 or 3|
# |'SCC'|  Calories consumption <br>monitoring| yes or no question|
# |'FAF'| Physical activity <br>frequency| FAF is between 0 to 3, 0 means no physical activity<br> and 3 means high workout. and again, in our data it's given as float|
# |'TUE'| Time using <br>technology devices| TUE is between 0 to 2. I think question will be "How long you have <br>been using technology devices to track your health." in our data it's given as float |
# |'CALC'| Consumption of alcohol | Takes 3 values: `Sometimes`, `no`, `Frequently`|
# | 'MTRANS' | Transportation used| MTRANS takes 5 values `Public_Transportation`, `Automobile`, <br>`Walking`, `Motorbike`, & `Bike`|
# |'NObeyesdad'| TARGET | This is our target, takes 7 values, and in this comp. we have to give <br>the class name (Not the Probability, which is the case in most comp.)
# 
# 
# <div style="font-size:120%"> 
#     <b>NObeyesdad (Target Variable):</b>
# </div>
# 
# * Insufficient_Weight : Less than 18.5
# * Normal_Weight       : 18.5 to 24.9
# * Obesity_Type_I      : 30.0 to 34.9
# * Obesity_Type_II     : 35.0 to 39.9
# * Obesity_Type_III   : Higher than 40
# * Overweight_Level_I, Overweight_Level_II takes values between 25 to 29
# 
# 

# # Import Libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders import OneHotEncoder, CatBoostEncoder, MEstimateEncoder
from sklearn.model_selection import StratifiedGroupKFold


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn import set_config
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
# import optuna
from sklearn.compose import ColumnTransformer
from prettytable import PrettyTable

from sklearn.compose import make_column_transformer
from sklearn.base import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
# import optuna
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Parameters

# Set Prameters for Reproduciblity
pd.set_option("display.max_rows",100)
# FILE_PATH = "../data/"
FILE_PATH= "./workspace/hyperopt/obesity-risk/data/"
TARGET = "NObeyesdad"
submission_path="ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

# # Load Data

# load all data
train = pd.read_csv(os.path.join(FILE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(FILE_PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(FILE_PATH, "sample_submission.csv"))
train_org = pd.read_csv(os.path.join(FILE_PATH, "ObesityDataSet.csv"))

# # Explore Data

def prettify_df(df):
    table = PrettyTable()
    table.field_names = df.columns

    for row in df.values:
        table.add_row(row)
    print(table)


train.head(10)

# Train Data
print("Train Data")
print(f"Total number of rows: {len(train)}")
print(f"Total number of columns: {train.shape[1]}\n")

# Test Data
print("Test Data")
print(f"Total number of rows: {len(test)}")
print(f"Total number of columns:{test.shape[1]}")

# check null and unique count
# FHWO: family_history_with_overweight
train_copy = train.rename(columns={"family_history_with_overweight":"FHWO"})
tmp = pd.DataFrame(index=train_copy.columns)
tmp['count'] = train_copy.count()
tmp['dtype'] = train_copy.dtypes
tmp['nunique'] = train_copy.nunique()
tmp['%nunique'] = (tmp['nunique']/len(train_copy))*100
tmp['%null'] = (train_copy.isnull().sum()/len(train_copy))*100
tmp['min'] = train_copy.min()
tmp['max'] = train_copy.max()
tmp

tmp.reset_index(inplace=True)
tmp = tmp.rename(columns = {"index":"Column Name"})
tmp = tmp.round(3)
prettify_df(tmp)
del tmp, train_copy

# Target Distribution with Gender

pd.set_option('display.float_format', '{:.2f}'.format)
tmp = pd.DataFrame(train.groupby([TARGET,'Gender'])["id"].agg('count'))
tmp.columns = ['Count']
train[TARGET].value_counts()
tmp = pd.merge(tmp,train[TARGET].value_counts(),left_index=True, right_index=True)
tmp.columns = ['gender_count','target_class_count']
tmp['%gender_count'] = tmp['gender_count']/tmp['target_class_count']
tmp["%target_class_count"] = tmp['target_class_count']/len(train) 
tmp = tmp[['gender_count','%gender_count','target_class_count','%target_class_count']]
print("Target Distribution with Gender")
tmp


raw_num_cols = list(train.select_dtypes("float").columns) 
raw_cat_cols = list(train.columns.drop(raw_num_cols+[TARGET]))

full_form = dict({'FAVC' : "Frequent consumption of high caloric food",
                  'FCVC' : "Frequency of consumption of vegetables",
                  'NCP' :"Number of main meal",
                  'CAEC': "Consumption of food between meals",
                  'CH2O': "Consumption of water daily",
                  'SCC':  "Calories consumption monitoring",
                  'FAF': "Physical activity frequency",
                  'TUE': "Time using technology devices",
                  'CALC': "Consumption of alcohol" ,
                  'MTRANS' : "Transportation used"})


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#PCA
pca = PCA(n_components=2)
pca_top_2 = pca.fit_transform(train[raw_num_cols])

tmp = pd.DataFrame(data = pca_top_2, columns = ['pca_1','pca_2'])
tmp['TARGET'] = train[TARGET]

#KMeans
kmeans = KMeans(7,random_state=RANDOM_SEED)
kmeans.fit(tmp[['pca_1','pca_2']])


def age_rounder(x):
    x_copy = x.copy()
    x_copy['Age'] = (x_copy['Age']*100).astype(np.uint16)
    return x_copy

def height_rounder(x):
    x_copy = x.copy()
    x_copy['Height'] = (x_copy['Height']*100).astype(np.uint16)
    return x_copy

def extract_features(x):
    x_copy = x.copy()
    x_copy['BMI'] = (x_copy['Weight']/x_copy['Height']**2)
#     x_copy['PseudoTarget'] = pd.cut(x_copy['BMI'],bins = [0,18.4,24.9,29,34.9,39.9,100],labels = [0,1,2,3,4,5],)    
    return x_copy

def col_rounder(x):
    x_copy = x.copy()
    cols_to_round = ['FCVC',"NCP","CH2O","FAF","TUE"]
    for col in cols_to_round:
        x_copy[col] = round(x_copy[col])
        x_copy[col] = x_copy[col].astype('int')
    return x_copy

AgeRounder = FunctionTransformer(age_rounder)
HeightRounder = FunctionTransformer(height_rounder)
ExtractFeatures = FunctionTransformer(extract_features)
ColumnRounder = FunctionTransformer(col_rounder)

# Using FeatureDropper we can drop columns. This is 
# important if we want to pass different set of features
# for different models

from sklearn.base import BaseEstimator, TransformerMixin
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self,x,y):
        return self
    def transform(self, x):
        return x.drop(self.cols, axis = 1)



# Encoding target values with int
target_mapping = {
                  'Insufficient_Weight':0,
                  'Normal_Weight':1,
                  'Overweight_Level_I':2,
                  'Overweight_Level_II':3, 
                  'Obesity_Type_I':4,
                  'Obesity_Type_II':5 ,
                  'Obesity_Type_III':6
                  }

# Define a method for Cross validation here we are using StartifiedKFold
skf = StratifiedKFold(n_splits=n_splits)

def cross_val_model(estimators,cv = skf, verbose = True):
    '''
        estimators : pipeline consists preprocessing, encoder & model
        cv : Method for cross validation (default: StratifiedKfold)
        verbose : print train/valid score (yes/no)
    '''
    
    X = train.copy()
    y = X.pop(TARGET)

    y = y.map(target_mapping)
    test_predictions = np.zeros((len(test),7))
    valid_predictions = np.zeros((len(X),7))

    val_scores, train_scores = [],[]
    for fold, (train_ind, valid_ind) in enumerate(skf.split(X,y)):
        model = clone(estimators)
        #define train set
        X_train = X.iloc[train_ind]
        y_train = y.iloc[train_ind]
        #define valid set
        X_valid = X.iloc[valid_ind]
        y_valid = y.iloc[valid_ind]

        model.fit(X_train, y_train)
        if verbose:
            print("-" * 100)
            print(f"Fold: {fold}")
            print(f"Train Accuracy Score:{accuracy_score(y_true=y_train,y_pred=model.predict(X_train))}")
            print(f"Valid Accuracy Score:{accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid))}")
            print("-" * 100)

        
        test_predictions += model.predict_proba(test)/cv.get_n_splits()
        valid_predictions[valid_ind] = model.predict_proba(X_valid)
        val_scores.append(accuracy_score(y_true=y_valid,y_pred=model.predict(X_valid)))
    if verbose: 
        print(f"Average Mean Accuracy Score: {np.array(val_scores).mean()}")
    return val_scores, valid_predictions, test_predictions

#Combine Orignal & Synthetic Data

train.drop(['id'],axis = 1, inplace = True)
test_ids = test['id']
test.drop(['id'],axis = 1, inplace=True)

train = pd.concat([train,train_org],axis = 0)
train = train.drop_duplicates()
train.reset_index(drop=True, inplace=True)

# empty dataframe to store score, & train / test predictions.
score_list, oof_list, predict_list = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# # Model

# <div style = "font-size:120%">Rather than focusing on a single model, in this competition it's better to combine predictions from many high performing models. In this notebook we will be training Four different type of models and will combine their predictions for final sub.</div>
# 
# * [Random Forest Model](#rfc)
# * [LGBM Model](#lgbm)
# * [XGB Model](#xgb)
# * [Catboost Model](#cat)
# 

# <a id = "rfC"> </a>
# # Random Forest Model

# Define Random Forest Model Pipeline

'''
RandomForestClassifier parameters: 


    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `"sqrt"`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
'''


RFC = make_pipeline(
                        ExtractFeatures,
                        MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                                           'SMOKE','SCC','CALC','MTRANS']),
                       RandomForestClassifier(random_state=RANDOM_SEED)
                    )

# Execute Random Forest Pipeline
val_scores,val_predictions,test_predictions = cross_val_model(RFC)

# Save train/test predictions in dataframes
for k,v in target_mapping.items():
    oof_list[f"rfc_{k}"] = val_predictions[:,v]

for k,v in target_mapping.items():
    predict_list[f"rfc_{k}"] = test_predictions[:,v]
# 0.8975337326149792
# 0.9049682643904575

# <a id = "lgbm"></a>
# # LGBM Model

# Define Optuna Function To Tune LGBM Model

# def lgbm_objective(trial):
#     params = {
#         'learning_rate' : trial.suggest_float('learning_rate', .001, .1, log = True),
#         'max_depth' : trial.suggest_int('max_depth', 2, 20),
#         'subsample' : trial.suggest_float('subsample', .5, 1),
#         'min_child_weight' : trial.suggest_float('min_child_weight', .1, 15, log = True),
#         'reg_lambda' : trial.suggest_float('reg_lambda', .1, 20, log = True),
#         'reg_alpha' : trial.suggest_float('reg_alpha', .1, 10, log = True),
#         'n_estimators' : 1000,
#         'random_state' : RANDOM_SEED,
#         'device_type' : "gpu",
#         'num_leaves': trial.suggest_int('num_leaves', 10, 1000),

#         #'boosting_type' : 'dart',
#     }
    
#     optuna_model = make_pipeline(
#                                  ExtractFeatures,
#                                  MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
#                                            'SMOKE','SCC','CALC','MTRANS']),
#                                 LGBMClassifier(**params,verbose=-1)
#                                 )
#     val_scores, _, _ = cross_val_model(optuna_model,verbose = False)
#     return np.array(val_scores).mean()

# lgbm_study = optuna.create_study(direction = 'maximize',study_name="LGBM")

# # Execute LGBM Tuning, To Tune set `TUNE` to True (it will take a long time)
# TUNE = False

# warnings.filterwarnings("ignore")
# if TUNE:
#     lgbm_study.optimize(lgbm_objective, 50)


numerical_columns = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')

# <div style = "font-size:120%">LGBM parameters in next cell are taken from @moazeldsokyx notebook you may check his great work in this notebook:<br></div>
# 
# https://www.kaggle.com/code/moazeldsokyx/pgs4e2-highest-score-lgbm-hyperparameter-tuning/notebook
# 

# Here we defined LGBM Pipeline
# Where we use One_Hot_Encoder, for categorical encoding
# standard scaler for numerical column scaling


params = {'learning_rate': 0.04325905707439143, 'max_depth': 4, 
          'subsample': 0.6115083405793659, 'min_child_weight': 0.43633356137010687, 
          'reg_lambda': 9.231766981717822, 'reg_alpha': 1.875987414096491, 'num_leaves': 373,
          'n_estimators' : 1000,'random_state' : RANDOM_SEED, 'device_type' : "gpu",
         }

best_params = {
    "objective": "multiclass",          # Objective function for the model
    "metric": "multi_logloss",          # Evaluation metric
    "verbosity": -1,                    # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",            # Gradient boosting type
    "random_state": 42,       # Random state for reproducibility
    "num_class": 7,                     # Number of classes in the dataset
    'learning_rate': 0.030962211546832760,  # Learning rate for gradient boosting
    'n_estimators': 500,                # Number of boosting iterations
    'lambda_l1': 0.009667446568254372,  # L1 regularization term
    'lambda_l2': 0.04018641437301800,   # L2 regularization term
    'max_depth': 10,                    # Maximum depth of the trees
    'colsample_bytree': 0.40977129346872643,  # Fraction of features to consider for each tree
    'subsample': 0.9535797422450176,    # Fraction of samples to consider for each boosting iteration
    'min_child_samples': 26             # Minimum number of data needed in a leaf
}

lgbm = make_pipeline(    
                        ColumnTransformer(
                        transformers=[('num', StandardScaler(), numerical_columns),
                                  ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                        LGBMClassifier(**best_params,verbose=-1)
                    )

# Train LGBM Model

val_scores,val_predictions,test_predictions = cross_val_model(lgbm)

for k,v in target_mapping.items():
    oof_list[f"lgbm_{k}"] = val_predictions[:,v]
    
for k,v in target_mapping.items():
    predict_list[f"lgbm_{k}"] = test_predictions[:,v]

#0.91420543252078

# <a id = "xgb"></a>
# # XGB Model

# Optuna study for XGB Model
# def xgb_objective(trial):
#     params = {
#         'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
#         'gamma' : trial.suggest_float('gamma', 1e-9, 1.0),
#         'subsample': trial.suggest_float('subsample', 0.25, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.25, 1.0),
#         'max_depth': trial.suggest_int('max_depth', 0, 24),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
#     }

#     params['booster'] = 'gbtree'
#     params['objective'] = 'multi:softmax'
#     params["device"] = "cuda"
#     params["verbosity"] = 0
#     params['tree_method'] = "gpu_hist"
    
    
#     optuna_model = make_pipeline(
# #                     ExtractFeatures,
#                     MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
#                                            'SMOKE','SCC','CALC','MTRANS']),
#                     XGBClassifier(**params,seed=RANDOM_SEED)
#                    )
    
#     val_scores, _, _ = cross_val_model(optuna_model,verbose = False)
#     return np.array(val_scores).mean()

# xgb_study = optuna.create_study(direction = 'maximize')


# # Tune using Optuna
# TUNE = False
# if TUNE:
#     xgb_study.optimize(xgb_objective, 50)

# XGB Pipeline

params = {
    'n_estimators': 1312,
    'learning_rate': 0.018279520260162645,
    'gamma': 0.0024196354156454324,
    'reg_alpha': 0.9025931173755949,
    'reg_lambda': 0.06835667255875388,
    'max_depth': 5,
    'min_child_weight': 5,
    'subsample': 0.883274050086088,
    'colsample_bytree': 0.6579828557036317
}
# {'eta': 0.018387615982905264, 'max_depth': 29, 'subsample': 0.8149303101087905, 'colsample_bytree': 0.26750463604831476, 'min_child_weight': 0.5292380065098192, 'reg_lambda': 0.18952063379457604, 'reg_alpha': 0.7201451827004944}

params = {'grow_policy': 'depthwise', 'n_estimators': 690, 
               'learning_rate': 0.31829021594473056, 'gamma': 0.6061120644431842, 
               'subsample': 0.9032243794829076, 'colsample_bytree': 0.44474031945048287,
               'max_depth': 10, 'min_child_weight': 22, 'reg_lambda': 4.42638097284094,
               'reg_alpha': 5.927900973354344e-07,'seed':RANDOM_SEED}

best_params = {'grow_policy': 'depthwise', 'n_estimators': 982, 
               'learning_rate': 0.050053726931263504, 'gamma': 0.5354391952653927, 
               'subsample': 0.7060590452456204, 'colsample_bytree': 0.37939433412123275, 
               'max_depth': 23, 'min_child_weight': 21, 'reg_lambda': 9.150224029846654e-08,
               'reg_alpha': 5.671063656994295e-08}
best_params['booster'] = 'gbtree'
best_params['objective'] = 'multi:softmax'
best_params["device"] = "cuda"
best_params["verbosity"] = 0
best_params['tree_method'] = "gpu_hist"
    
XGB = make_pipeline(
#                     ExtractFeatures,
#                     MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
#                                            'SMOKE','SCC','CALC','MTRANS']),
#                     FeatureDropper(['FAVC','FCVC']),
#                     ColumnRounder,
#                     ColumnTransformer(
#                     transformers=[('num', StandardScaler(), numerical_columns),
#                                   ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)]),
                    MEstimateEncoder(cols=['Gender','family_history_with_overweight','FAVC','CAEC',
                                           'SMOKE','SCC','CALC','MTRANS']),
                    XGBClassifier(**best_params,seed=RANDOM_SEED)
                   )

val_scores,val_predictions,test_predictions = cross_val_model(XGB)

for k,v in target_mapping .items():
    oof_list[f"xgb_{k}"] = val_predictions[:,v]

for k,v in target_mapping.items():
    predict_list[f"xgb_{k}"] = test_predictions[:,v]
    
# 0.90634942296329
#0.9117093455898445 with rounder
#0.9163506382522121

# <a id = "cat"></a>
# # Catboost Model
# 

# Optuna Function For Catboost Model
# def cat_objective(trial):
    
#     params = {
        
#         'iterations': 1000,  # High number of estimators
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'depth': trial.suggest_int('depth', 3, 10),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 10.0),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'random_seed': RANDOM_SEED,
#         'verbose': False,
#         'task_type':"GPU"
#     }
    
#     cat_features = ['Gender','family_history_with_overweight','FAVC','FCVC','NCP',
#                 'CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
#     optuna_model = make_pipeline(
#                         ExtractFeatures,
# #                         AgeRounder,
# #                         HeightRounder,
# #                         MEstimateEncoder(cols = raw_cat_cols),
#                         CatBoostClassifier(**params,cat_features=cat_features)
#                         )
#     val_scores,_,_ = cross_val_model(optuna_model,verbose = False)
#     return np.array(val_scores).mean()
    
# cat_study = optuna.create_study(direction = 'maximize')

params = {'learning_rate': 0.13762007048684638, 'depth': 5, 
          'l2_leaf_reg': 5.285199432056192, 'bagging_temperature': 0.6029582154263095,
         'random_seed': RANDOM_SEED,
        'verbose': False,
        'task_type':"GPU",
         'iterations':1000}

cat_features_indices = [train.columns.get_loc(col) for col in categorical_columns]

CB = make_pipeline(
    MEstimateEncoder(cols=categorical_columns),
    CatBoostClassifier(**params, cat_features=categorical_columns)
)
# CB = make_pipeline(
# #                         ExtractFeatures,
# #                         AgeRounder,
# #                         HeightRounder,
# #                         MEstimateEncoder(cols = raw_cat_cols),
# #                         CatBoostEncoder(cols = cat_features),
#                         CatBoostClassifier(**params, cat_features=categorical_columns)
#                         )

# Train Catboost Model
val_scores,val_predictions,test_predictions = cross_val_model(CB)
for k,v in target_mapping.items():
    oof_list[f"cat_{k}"] = val_predictions[:,v]

for k,v in target_mapping.items():
    predict_list[f"cat_{k}"] = test_predictions[:,v]

# best 0.91179835368868 with extract features, n_splits = 10
# best 0.9121046227778054 without extract features, n_splits = 10

# # Model Evaluation

# skf = StratifiedKFold(n_splits=5)
weights = {"rfc_":0,
           "lgbm_":3,
           "xgb_":1,
           "cat_":0}
tmp = oof_list.copy()
for k,v in target_mapping.items():
    tmp[f"{k}"] = (weights['rfc_']*tmp[f"rfc_{k}"] +
              weights['lgbm_']*tmp[f"lgbm_{k}"]+
              weights['xgb_']*tmp[f"xgb_{k}"]+
              weights['cat_']*tmp[f"cat_{k}"])    
tmp['pred'] = tmp[target_mapping.keys()].idxmax(axis = 1)
tmp['label'] = train[TARGET]
print(f"Ensemble Accuracy Scoe: {accuracy_score(train[TARGET],tmp['pred'])}")
    
cm = confusion_matrix(y_true = tmp['label'].map(target_mapping),
                      y_pred = tmp['pred'].map(target_mapping),
                     normalize='true')

cm = cm.round(2)
# plt.figure(figsize=(8,8))
# disp = ConfusionMatrixDisplay(confusion_matrix = cm,
#                               display_labels = target_mapping.keys())
# disp.plot(xticks_rotation=50)
# plt.tight_layout()
# plt.show()

"""   BEST     """

# Best LB [0,1,0,0]
# Average Train Score:0.9142044335854003
# Average Valid Score:0.91420543252078

# Best CV [1,3, 1,1]
# Average Train Score:0.9168308163711971
# Average Valid Score:0.9168308163711971
# adding orignal data improves score

# # Final Submission

for k,v in target_mapping.items():
    predict_list[f"{k}"] = (weights['rfc_']*predict_list[f"rfc_{k}"]+
                            weights['lgbm_']*predict_list[f"lgbm_{k}"]+
                            weights['xgb_']*predict_list[f"xgb_{k}"]+
                            weights['cat_']*predict_list[f"cat_{k}"])

final_pred = predict_list[target_mapping.keys()].idxmax(axis = 1)

sample_sub[TARGET] = final_pred
sample_sub.to_csv(os.path.join(FILE_PATH,submission_path),index=False)


score= 1-accuracy_score(train[TARGET],tmp['pred'])
