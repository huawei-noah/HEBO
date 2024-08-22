#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

FILE_PATH = "../data/"
# FILE_PATH= "./workspace/hyperopt/bsd2/data/"

submission_path = "ori_submission.csv"
RANDOM_SEED = 73

# Load datasets
bike_train = pd.read_csv(FILE_PATH + 'train.csv', parse_dates=['datetime'])
bike_test = pd.read_csv(FILE_PATH + 'test.csv', parse_dates=['datetime'])

# Combine the datasets for consistent preprocessing
combine = [bike_train, bike_test]

# Convert the datetime column to datetime and create new features
for dataset in combine:
    dataset['datetime'] = pd.to_datetime(dataset['datetime'])
    dataset['day_of_week'] = dataset['datetime'].dt.dayofweek
    dataset['month'] = dataset['datetime'].dt.month
    dataset['hour'] = dataset['datetime'].dt.hour

# Create copies for further processing
bike_copy = bike_train.copy()
season_dict = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
weather_dict = {1: "Clear", 2: "Mist", 3: "Light Snow", 4: "Heavy Rain"}

# Replace numerical values with categorical names
for dataset in combine:
    if 'season' in dataset.columns:
        dataset['season'].replace(season_dict, inplace=True)
    if 'weather' in dataset.columns:
        dataset['weather'].replace(weather_dict, inplace=True)

# Log transformation for skewed features
for dataset in combine:
    for col in ['windspeed', 'casual', 'registered']:
        if col in dataset.columns:
            dataset[col] = np.log1p(dataset[col])  # Use log1p to handle log(0)

# Outlier removal function
def remove_outliers(dataframe, column_name, threshold=0.3):
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' doesn't exist")
        return dataframe
    while True:
        Q1 = dataframe[column_name].quantile(0.25)
        Q3 = dataframe[column_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = dataframe[((dataframe[column_name] < (Q1 - 1.5 * IQR)) | 
                              (dataframe[column_name] > (Q3 + 1.5 * IQR)))]
        percentage_outliers = len(outliers) / len(dataframe)
        if percentage_outliers > threshold:
            print(f"Percentage of outliers in {column_name} exceeds the threshold! Can't remove outliers")
            return dataframe
        if len(outliers) == 0:
            break
        dataframe.drop(outliers.index, inplace=True)
    return dataframe

# Remove outliers in the combined datasets

for col in ['windspeed', 'casual', 'registered']:
    if col in bike_train.columns:
        remove_outliers(dataframe=bike_train, column_name=col, threshold=0.3)

# One-hot encoding for categorical features
bike_train = pd.get_dummies(bike_train, columns=['season', 'weather'], drop_first=True)
bike_test = pd.get_dummies(bike_test, columns=['season', 'weather'], drop_first=True)

# Drop unnecessary columns and create high correlation features for modeling
for dataset in combine:
    dataset.drop(['datetime', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity'], axis=1, inplace=True, errors='ignore')

x = bike_train.drop(['count', 'casual', 'registered', 'datetime'], axis=1, errors='ignore')
y = bike_train['count']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

# Models for evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# Evaluate models and find the best one
best_model = None
best_score = float('-inf')

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (name, model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'{name} - Mean Squared Error: {mse:.2f}, R2 Score: {r2:.2f}')
    
    if r2 > best_score:
        best_score = r2
        best_model = pipeline


# # Final model training on full train dataset with the best model
# best_model.fit(x, y)

# # Preprocess the test dataset similarly
# X_test = bike_test.drop(['datetime'], axis=1, errors='ignore')
# y_test_pred = best_model.predict(X_test)

# # Prepare the submission
# submission = pd.DataFrame({'datetime': bike_test['datetime'], 'count': (y_test_pred).astype(int)})
# submission.to_csv(submission_path, index=False)



# score=1.0-best_score
