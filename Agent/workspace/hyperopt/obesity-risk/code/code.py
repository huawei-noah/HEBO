#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# This project focuses on predicting the risk of obesity using a machine learning model. The solution presented here utilizes LightGBM (LGBM) combined with Optuna for hyperparameter optimization. It is part of the "Playground Series - Season 4, Episode 2" by Aryangupta30, available on Kaggle.
# 
# ## Description
# Hyperparameter tuning plays a crucial role in enhancing the performance of machine learning models. This project demonstrates the effectiveness of using Optuna to systematically search for the best set of hyperparameters for the LGBM model.
# 
# ### Methodology
# The Optuna module is employed to search through a predefined set of hyperparameters and a specified number of random parameter combinations. The hyperparameters considered for tuning include [list of hyperparameters].
# 
# ### Impact on Model Performance
# After hyperparameter tuning, the model"s accuracy increased to 91.943%, representing a 0.903% improvement over the previous performance. This enhancement underscores the importance of optimizing hyperparameters to fine-tune the model"s behavior and improve its predictive power.
# 
# ## Conclusion
# Incorporating hyperparameter tuning has resulted in a substantial improvement in model performance, as evidenced by the increase in accuracy. This highlights the significance of fine-tuning model parameters to achieve optimal results in machine learning tasks.
# 
# ## Notebooks and Code
# The project notebooks can be found on Kaggle at [Divyam6969"s profile](www.kaggle.com/divyam6969).
# 
# ## Code Overview
# The project involves the following key steps:
# 
# 1. Importing necessary libraries including pandas, seaborn, matplotlib, numpy, scikit-learn, LightGBM, and Optuna.
# 2. Loading and initial inspection of the datasets including train_data, test_data, and original_data.
# 3. Visualizing null values in the datasets (there are no null values).
# 4. Extracting variable types (continuous and categorical variables).
# 5. Plotting distribution of categorical columns.
# 6. Plotting histograms and density plots for continuous variables.
# 7. Utilizing Optuna for hyperparameter optimization.
# 8. Adjusting hyperparameters based on performance evaluation.
# 9. Creating an instance of LGBMClassifier with the best parameters.
# 10. Making predictions on the test set and generating a submission file.
# 
# ## Final Submission
# The final predictions are made using the optimized model and submitted to the competition. The submission achieves a high accuracy score.
# 
# ## Additional Information
# Further optimization of the model can be explored by adjusting additional hyperparameters. If you found this notebook helpful, please consider upvoting it.
# 
# 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won"t be saved outside of the current session

# # Importing libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importing scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Importing LGBMClassifier
from lightgbm import LGBMClassifier

# Importing Optuna for hyperparameter optimization
# from optuna.samplers import TPESampler
# import optuna

# Ignoring warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")

# Pandas setting to display more dataset rows and columns
pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 600)

# Set Seaborn style
sns.set(style="whitegrid")

# train_data = pd.read_csv("./workspace/hyperopt/obesity-risk/data/train.csv")
train_data = pd.read_csv("../data/train.csv")
train_data.name = "Train Dataset"

# test_data = pd.read_csv("./workspace/hyperopt/obesity-risk/data/test.csv")
test_data = pd.read_csv("../data/test.csv")

# sample_submission = pd.read_csv("./workspace/hyperopt/obesity-risk/data/sample_submission.csv")
sample_submission = pd.read_csv("../data/sample_submission.csv")

# original_data = pd.read_csv("./workspace/hyperopt/obesity-risk/data/ObesityDataSet.csv")
original_data = pd.read_csv("../data/ObesityDataSet.csv")
original_data.name = "Original Dataset"

# print("# Train Data INFO\n")
# print(train_data.info())
# print("="*50)
# print("\n# Original Data INFO\n")
# print(train_data.info())

train_data.head()

train_data.describe().T.style.background_gradient()

test_data.head()

test_data.describe().T.style.background_gradient()

original_data.head()

original_data.describe().T.style.background_gradient()


# # Create a subplot with dimensions (1, 3)
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # Visualize null values in train dataset
# sns.heatmap(train_data.isna(), cmap="gray", cbar=False, ax=axes[0])
# axes[0].set_title("Train Dataset")

# # Visualize null values in test dataset
# sns.heatmap(test_data.isna(), cmap="gray", cbar=False, ax=axes[1])
# axes[1].set_title("Test Dataset")

# # Visualize null values in original dataset
# sns.heatmap(original_data.isna(), cmap="gray", cbar=False, ax=axes[2])
# axes[2].set_title("Original Dataset")

# plt.show()

def get_variable_types(dataframe):
    continuous_vars = []
    categorical_vars = []

    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            categorical_vars.append(column)
        else:
            continuous_vars.append(column)

    return continuous_vars, categorical_vars


continuous_vars, categorical_vars = get_variable_types(train_data)
continuous_vars.remove("id"), categorical_vars.remove("NObeyesdad")


# print("Continuous Variables:", continuous_vars)
# print("Categorical Variables:", categorical_vars)


def plot_distribution(dataframe, target_column):
    # Calculate value counts
    value_counts = dataframe[target_column].value_counts()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar plot on the first subplot
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis", ax=ax1)
    ax1.set_xlabel(target_column, fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # Add data labels above each bar
    for index, value in enumerate(value_counts):
        ax1.text(index, value, str(value), ha="center", va="bottom", fontsize=10)

    # Pie plot on the second subplot
    ax2.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%",
            colors=sns.color_palette("viridis", len(value_counts)))
    ax2.axis("equal")

    # Main title for the figure
    fig.suptitle(f"Comparison of {target_column} Distribution in ({dataframe.name})", fontsize=18)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


# plot_distribution(train_data, "NObeyesdad")

# plot_distribution(original_data, "NObeyesdad")

# for column in categorical_vars:
#     plot_distribution(train_data, column)

# def plot_histograms_and_density(dataframe, columns):
#     for column in columns:
#         fig, ax = plt.subplots(figsize=(16, 4))
#         fig = sns.histplot(data=train_data, x=column, hue="NObeyesdad", bins=50, kde=True)
#         plt.ylim(0,500)
#         plt.show()

# plot_histograms_and_density(train_data, continuous_vars)

train = pd.concat([train_data, original_data]).drop(["id"], axis=1).drop_duplicates()
test = test_data.drop(["id"], axis=1)

train = pd.get_dummies(train,
                       columns=categorical_vars)
test = pd.get_dummies(test,
                      columns=categorical_vars)

X = train.drop(["NObeyesdad"], axis=1)
y = train["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing datasets
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

# #The tuning process has been commented out due to its time-consuming nature.

# # Define the objective function for Optuna optimization
# def objective(trial, X_train, y_train, X_test, y_test):
#     # Define parameters to be optimized for the LGBMClassifier
#     param = {
#         "objective": "multiclass",
#         "metric": "multi_logloss",
#         "verbosity": -1,
#         "boosting_type": "gbdt",
#         "random_state": 42,
#         "num_class": 7,
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
#         "n_estimators": trial.suggest_int("n_estimators", 400, 600),
#         "lambda_l1": trial.suggest_float("lambda_l1", 0.005, 0.015),
#         "lambda_l2": trial.suggest_float("lambda_l2", 0.02, 0.06),
#         "max_depth": trial.suggest_int("max_depth", 6, 14),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
#         "subsample": trial.suggest_float("subsample", 0.8, 1.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
#     }

#     # Create an instance of LGBMClassifier with the suggested parameters
#     lgbm_classifier = LGBMClassifier(**param)

#     # Fit the classifier on the training data
#     lgbm_classifier.fit(X_train, y_train)

#     # Evaluate the classifier on the test data
#     score = lgbm_classifier.score(X_test, y_test)

#     return score

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust the test_size as needed

# # Set up the sampler for Optuna optimization
# sampler = optuna.samplers.TPESampler(seed=42)  # Using Tree-structured Parzen Estimator sampler for optimization

# # Create a study object for Optuna optimization
# study = optuna.create_study(direction="maximize", sampler=sampler)

# # Run the optimization process
# study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)

# # Get the best parameters after optimization
# best_params = study.best_params

# print("="*50)
# print(best_params)

# Best parameters obtained from Optuna optimization process

best_params = {
    "objective": "multiclass",  # Objective function for the model
    "metric": "multi_logloss",  # Evaluation metric
    "verbosity": -1,  # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",  # Gradient boosting type
    "random_state": 42,  # Random state for reproducibility
    "num_class": 7,  # Number of classes in the dataset
    "learning_rate": 0.030962211546832760,  # Learning rate for gradient boosting
    "n_estimators": 500,  # Number of boosting iterations
    "lambda_l1": 0.009667446568254372,  # L1 regularization term
    "lambda_l2": 0.04018641437301800,  # L2 regularization term
    "max_depth": 10,  # Maximum depth of the trees
    "colsample_bytree": 0.40977129346872643,  # Fraction of features to consider for each tree
    "subsample": 0.9535797422450176,  # Fraction of samples to consider for each boosting iteration
    "min_child_samples": 26  # Minimum number of data needed in a leaf
}

# w/o comment 
best_params["learning_rate"] = 0.04644598811864853
best_params["n_estimators"] = 555
best_params["lambda_l1"] = 0.014203698221554291
best_params["lambda_l2"] = 0.045788523601265375
best_params["max_depth"] = 12
best_params["colsample_bytree"] = 0.5643565110240454
best_params["subsample"] = 0.9833762016567981
best_params["min_child_samples"] = 14

best_params = {
    "objective": "multiclass",  # Objective function for the model
    "metric": "multi_logloss",  # Evaluation metric
    "verbosity": -1,  # Verbosity level (-1 for silent)
    "boosting_type": "gbdt",  # Gradient boosting type
    "random_state": 42,  # Random state for reproducibility
    "num_class": 7,  # Number of classes in the dataset
    "learning_rate": 0.02182166572600718,  # Learning rate for gradient boosting
    "n_estimators": 553,  # Number of boosting iterations
    "lambda_l1": 0.009083985522829635,  # L1 regularization term
    "lambda_l2": 0.029567543144150783,  # L2 regularization term
    "max_depth": 7,  # Maximum depth of the trees
    "colsample_bytree": 0.5389636874259516,  # Fraction of features to consider for each tree
    "subsample": 0.8635557253963909,  # Fraction of samples to consider for each boosting iteration
    "min_child_samples": 47  # Minimum number of data needed in a leaf
}

lgbm_classifier = LGBMClassifier(**best_params)

# lgbm_classifier.fit(X_train, y_train)

# y_pred = lgbm_classifier.predict(X_test)
# accuracy=accuracy_score(y_test, y_pred) 


# predictions = lgbm_classifier.predict(test)
# predictions

# sample_submission["NObeyesdad"] = predictions
# sample_submission

# # sample_submission.to_csv("./workspace/hyperopt/obesity-risk/data/submission_wocomment.csv", index=False)
# sample_submission.to_csv("../data/submission_wcomment.csv", index=False)

# #0.084482 bo
# #0.08316918362880277 ori
# score = 1.0 - accuracy
