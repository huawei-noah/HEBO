import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import optuna
import lightgbm as lgbm
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

train_data = pd.read_csv("./workspace/hyperopt/data/train.csv", index_col=0)
n = train_data.shape[0]
train, val = train_data.iloc[:int(n * 0.8)], train_data.iloc[int(n * 0.8):]
train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
# test = pd.read_csv("./workspace/hyperopt/data/test.csv", index_col=0)
Expenses_columns = ['RoomService', 'FoodCourt', 'Spa', 'VRDeck', 'ShoppingMall']


def features(data):
    data.loc['Age_group'] = 0
    data.loc[(train['Age'] > 0) & (data['Age'] <= 5), 'Age_group'] = 1
    data.loc[(train['Age'] > 5) & (data['Age'] <= 10), 'Age_group'] = 2
    data.loc[(train['Age'] > 10) & (data['Age'] <= 20), 'Age_group'] = 3
    data.loc[(train['Age'] > 20) & (data['Age'] <= 30), 'Age_group'] = 4
    data.loc[(train['Age'] > 30) & (data['Age'] <= 50), 'Age_group'] = 5
    data.loc[(train['Age'] > 50) & (data['Age'] <= 60), 'Age_group'] = 6
    data.loc[(train['Age'] > 60) & (data['Age'] <= 70), 'Age_group'] = 7
    data.loc[(train['Age'] > 70) & (data['Age'] <= 100), 'Age_group'] = 8
    data.Age_group = data.Age_group.astype(float)

    data['RoomService'] = np.where(data['CryoSleep'] == True, 0, data['RoomService'])
    data['FoodCourt'] = np.where(data['CryoSleep'] == True, 0, data['FoodCourt'])
    data['ShoppingMall'] = np.where(data['CryoSleep'] == True, 0, data['ShoppingMall'])
    data['Spa'] = np.where(data['CryoSleep'] == True, 0, data['Spa'])
    data['VRDeck'] = np.where(data['CryoSleep'] == True, 0, data['VRDeck'])

    data['Group'] = data['PassengerId'].astype(str).str[:4].astype(float)
    data[['Deck', 'Number', 'Side']] = data['Cabin'].str.split('/', expand=True)

    data['Expenses'] = data.loc[:, Expenses_columns].sum(axis=1)
    data.loc[:, ['CryoSleep']] = data.apply(lambda x: True if x.Expenses == 0 and pd.isna(x.CryoSleep) else x, axis=1)
    data['VIP'] = np.where(data['CryoSleep'] == 0, True, False).astype(object)

    data['Name'] = data['Name'].fillna('Unknown Unknown')
    data.loc[:, ['Surname']] = data.Name.str.split(expand=True)[1]

    return data


train = features(train.copy())
val = features(val.copy())

train = train.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
val = val.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
X_train = train.drop('Transported', axis=1)
y_train = train.Transported.astype('int')
X_val = val.drop('Transported', axis=1)
y_val = val.Transported.astype('int')

# Missing Values
imputer = KNNImputer(n_neighbors=5)
num_features = X_train.select_dtypes('float64').columns.to_list()
cat_features = X_train.select_dtypes('object').columns.to_list()


def missing(data):
    data[cat_features] = data[cat_features].infer_objects(copy=False).fillna('None').astype('category')
    data[num_features] = imputer.fit_transform(data[num_features])
    return  data


X_train = missing(X_train)
X_val = missing(X_val)
val_preds = []
results = []

# @MODEL_START@
cat = CatBoostClassifier(learning_rate=0.1, depth=13, l2_leaf_reg=0.3, random_state=42, eval_metric='Accuracy', iterations=150, early_stopping_rounds=20)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
cat.fit(train_pool)
# @MODEL_END@

y_pred_val = cat.predict(X_val)
accuracy = (y_pred_val == np.array(y_val)).sum() / len(y_val)
print("ACCURACY:", accuracy)

























