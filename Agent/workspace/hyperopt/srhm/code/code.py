#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# FILE_PATH = "../data/"
FILE_PATH = "./workspace/hyperopt/srhm/data/"

submission_path = "best_submission.csv"
RANDOM_SEED = 73

train_df = pd.read_csv(FILE_PATH + "train.csv", parse_dates=["timestamp"])
dtype_df = train_df.dtypes.reset_index()
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["qtd_atributos", "tipo"]
dtype_df.groupby("tipo").aggregate("count").reset_index()

# * Transformar as colunas categóricas para rodar o XGBoost e ver as colunas mais relevantes.
# * Usar ordinal encoding porque já tem muitas colunas e o objetivo é apenas filtrar algumas delas

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

cat = [a for a in train_df.columns if train_df[a].dtype == "object"]
train_df[cat] = encoder.fit_transform(train_df[cat])
dtype_df = train_df.dtypes.reset_index()
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["qtd_atributos", "tipo"]
dtype_df.groupby("tipo").aggregate("count").reset_index()

# * Verificar a proporção de dados faltantes em cada coluna

missing = train_df.isnull().sum() * 100 / len(train_df)
missing_df = pd.DataFrame({"col": train_df.columns, "missing_values": missing})
missing_df = missing_df[missing_df.missing_values != 0]
missing_df.sort_values("missing_values", inplace=True, ascending=False)
missing_df.head(10)

missing_df = missing_df[missing_df.missing_values > 40]
cols_to_remove = missing_df.col.to_list()
train_df.drop(cols_to_remove, inplace=True, axis=1)

from sklearn.impute import SimpleImputer

# removendo coluna tipo datetime
train_df.drop("timestamp", inplace=True, axis=1)

imputer = SimpleImputer(strategy="most_frequent")
itrain_df = pd.DataFrame(imputer.fit_transform(train_df))
itrain_df.columns = train_df.columns

from xgboost import XGBRegressor

X = itrain_df.drop("price_doc", axis=1)
y = itrain_df["price_doc"]

model = XGBRegressor(n_estimators=200, max_depth=13, random_state=987, eta=0.01)
model.fit(X, y)

feature_importance_df = pd.DataFrame({"col": X.columns, "importance": model.feature_importances_})
feature_importance_df.sort_values("importance", inplace=True, ascending=False)
feature_importance_df.head(10)

cols = feature_importance_df[feature_importance_df.importance > 0.016]
train = itrain_df[cols.col.to_list()]
train["price_doc"] = itrain_df.price_doc.values
train.head()

# * Alta correlação entre os atributos que dizem respeito à área em que o imóvel está localizado, com exceção de "ttk_km".
# * Avaliar atributos a serem removidos com base nas correlações entre eles e com "price_doc".

cols_to_remove = ["cafe_count_3000_price_2500",
                  "cafe_count_5000_price_2500",
                  "cafe_count_2000",
                  "office_sqm_5000",
                  "cafe_count_1500_price_high"]

cols = ["id", "full_sq", "culture_objects_top_25", "female_f", "build_count_monolith", "cafe_count_3000",
        "sport_count_3000", "price_doc"]
df_train = pd.read_csv(FILE_PATH + "train.csv", usecols=cols)
cols.remove("price_doc")
df_test = pd.read_csv(FILE_PATH + "test.csv", usecols=cols)

df_train["build_count_monolith"].value_counts()

df_train["build_count_monolith"].replace(np.NaN, 1.0, inplace=True)
df_test["build_count_monolith"].replace(np.NaN, 1.0, inplace=True)

df_train["culture_objects_top_25"].unique()

df_train["bin_culture_objects_top_25"] = np.where(df_train["culture_objects_top_25"] == "yes", 1, 0)
df_train.drop("culture_objects_top_25", axis=1, inplace=True)
df_test["bin_culture_objects_top_25"] = np.where(df_test["culture_objects_top_25"] == "yes", 1, 0)
df_test.drop("culture_objects_top_25", axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error

X = df_train.drop(["id", "price_doc"], axis=1)
y = np.log(df_train.price_doc)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=9)
xgb = XGBRegressor(n_estimators=286, max_depth=9, random_state=9, eta=.03)
# xgb.fit(x_train, y_train)
# y_pred = xgb.predict(x_test)
# # print("RMSLE: ", mean_squared_log_error(y_test,y_pred,squared=False))


# test_ids = df_test.id.values
# test_data = df_test.drop("id", axis=1)
# predictions = np.exp(xgb.predict(test_data))
# sub_preview = pd.DataFrame({"id": test_ids, "price_doc": predictions})
# sub_preview.head()


# submission = pd.read_csv(FILE_PATH+"sample_submission.csv")
# submission["price_doc"] = predictions
# submission.to_csv(submission_path, index=False)
# submission.head()

# score= mean_squared_log_error(y_test,y_pred,squared=False)
