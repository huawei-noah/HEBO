#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler



# FILE_PATH= "../data/"
FILE_PATH= "./workspace/hyperopt/sf-crime/data/"

submission_path="ori_submission.csv"
RANDOM_SEED = 73



df = pd.read_csv(FILE_PATH+'train.csv.zip', parse_dates=['Dates'])
# df = pd.read_csv(FILE_PATH+'train.csv.zip')




# - Dates - timestamp of the crime incident
# - Category - category of the crime incident (only in train.csv). This is the target variable you are going to predict.
# - Descript - detailed description of the crime incident (only in train.csv)
# - DayOfWeek - the day of the week
# - PdDistrict - name of the Police Department District
# - Resolution - how the crime incident was resolved (only in train.csv)
# - Address - the approximate street address of the crime incident 
# - X - Longitude
# - Y - Latitude
print(df.columns)
# print(df.Category.nunique())
df.Category.value_counts(normalize=True)[:10]


# Целевая метка - явный дисбаланс классов

usecols = ['Dates', 'X','Y', 'Category']
df.shape, df.drop_duplicates(subset=usecols).shape


# df[df.duplicated(subset=usecols)].sort_values(usecols)
# неполные дубликаты, разные преступления


print(df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)


# Проверка гео

df['X'].describe()


df['Y'].describe()


df[df.Y > 38].shape


df[df.Y > 38].duplicated(subset=['X', 'Y', 'Category'], keep=False).sum()


# явная ошибка, не восстановить, убираю
df = df[df.Y <= 38].reset_index(drop=True)


# sns.scatterplot(
#     data=df.drop_duplicates(subset=['X','Y', 'Category']), 
#     x='X', 
#     y='Y', 
#     hue='Category'
# )
# plt.legend(labels = []);


# # Визуально преступления не зависят от района, распределены равномерно

# sns.scatterplot(
#     data=df.drop_duplicates(subset=['X','Y', 'Category', 'PdDistrict']), 
#     x='X', 
#     y='Y', 
#     hue='PdDistrict'
# )
# plt.legend(labels = []);


# Несмотря на наличие четких границ по полицейским участкам, есть выбивающиеся значения

# df['target']
df.Category.value_counts()


df['year'] = df['Dates'].dt.year
df['month'] = df['Dates'].dt.month
df['day'] = df['Dates'].dt.day

df['dayofweek'] = df['Dates'].dt.dayofweek


usecols = [
    'PdDistrict', 'Address', 'X', 'Y', 
    'year', 'month', 'day', 'dayofweek'
]


# Кодирование признаков

df.Descript.nunique(), df.PdDistrict.nunique(), df.Address.nunique()


df[usecols].duplicated().sum(), df[usecols + ['Category']].duplicated().sum()


df = df.drop_duplicates(subset=usecols).reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(
    df[usecols], 
    df.Category, 
    test_size=.3, 
    random_state=2024, 
    shuffle=True, 
)

X_train.shape, X_test.shape


cat_features = ['PdDistrict', 'Address', 'dayofweek']


cat_pipe = Pipeline([
    ('encoder', OrdinalEncoder(
        unknown_value=-1, 
        handle_unknown='use_encoded_value'))
])

rest_pipe = Pipeline([
    ('scaler', StandardScaler())
])


prep_pipe = ColumnTransformer([
    ('cat_pipe', cat_pipe, cat_features),
    ('rest_pipe', rest_pipe, [x for x in X_train.columns if x not in cat_features])
])


prep_pipe.fit(X_train, y_train)


prep_pipe.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# Подбор количества итераций

x = []
test_scores = []
train_scores = []
X_test_transformed = prep_pipe.transform(X_test)
X_train_transformed = prep_pipe.transform(X_train)

for n in range(1, 30):
    # print(f'step {n}', end='\r')
    clf = RandomForestClassifier(n_estimators=n, n_jobs=-1, max_depth=3)
    clf.fit(X_train_transformed, y_train)
    
    test_scores.append(
    roc_auc_score(
        y_test, 
        clf.predict_proba(X_test_transformed), 
        multi_class='ovr'
    ))

    train_scores.append(
    roc_auc_score(
        y_train, 
        clf.predict_proba(X_train_transformed), 
        multi_class='ovr'
    ))
    
    x.append(n)
    
    if abs(test_scores[-1] - train_scores[-1]) > .2:
        print(f'n_estimators: {n}')
        print(f'scores train/test {train_scores[-1]:.2f}/{test_scores[-1]:2f}')
        break


# plt.plot(x, train_scores)
# plt.plot(x, test_scores)
# plt.grid()
# plt.title('ROC AUC by n estimators')
# plt.xlabel('estimators count')
# plt.ylabel('ROC AUC');


n_estimator = [i for i,x in enumerate(test_scores) if x == max(test_scores)]



clf = RandomForestClassifier(n_estimators=n, n_jobs=-1, max_depth=3)
# clf.fit(X_train_transformed, y_train)


# # test = pd.read_csv(FILE_PATH+'test.csv.zip', parse_dates=['Dates'])


# # test['year'] = test.Dates.dt.year
# # test['month'] = test.Dates.dt.month
# # test['day'] = test.Dates.dt.day
# # test['dayofweek'] = test.Dates.dt.dayofweek


# # result = pd.DataFrame(
# #     clf.predict_proba(prep_pipe.transform(test[usecols])), 
# #     columns = clf.classes_
# # )


# # result['Id'] = test['Id']


# # result.to_csv(FILE_PATH+'submission.csv', index=False)

# import numpy  as np
# score= 1.0 -np.mean(test_scores)

