import csv
import math
import os
import random
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#########################################################################
# Metrics 
#########################################################################


FILE_PATH = "./workspace/hyperopt/higgs-boson2/data/"
# FILE_PATH ="../data/"
TARGET = "NObeyesdad"
submission_path = "best_submission.csv"
RANDOM_SEED = 73


def ams(s, b):
    return math.sqrt(2 * ((s + b + 10) * math.log(1.0 + s / (b + 10)) - s))


def get_ams_score(W, Y, Y_pred):
    s = W * (Y == 1) * (Y_pred == 1)
    b = W * (Y == 0) * (Y_pred == 1)
    s = np.sum(s)
    b = np.sum(b)
    return ams(s, b)


def ams_scorer(W, Y, Y_pred_proba):
    Y_pred = Y_pred_proba[:, 1] > 0.5  # Thresholding at 0.5 for AMS calculation
    return get_ams_score(W, Y, Y_pred)


#########################################################################
# Models 
#########################################################################


def nested_model():
    estimator = ExtraTreesClassifier(
        n_estimators=400,
        max_features=30,
        max_depth=12,
        min_samples_leaf=100,
        min_samples_split=100,
        verbose=1,
        n_jobs=-1)
    classifier = AdaBoostClassifier(
        n_estimators=20,
        learning_rate=0.75,
        estimator=estimator
    )
    return classifier


#########################################################################
# Feature preprocessing
#########################################################################

def preprocess(X, X_test):
    # Impute missing data.
    imputer = SimpleImputer(missing_values=-999.0, strategy="most_frequent")
    X = imputer.fit_transform(X)
    X_test = imputer.transform(X_test)

    # Create inverse log values of features which are positive in value.
    inv_log_cols = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 16, 19, 21, 23, 26)
    X_inv_log_cols = np.log(1 / (1 + X[:, inv_log_cols]))
    X = np.hstack((X, X_inv_log_cols))
    X_test_inv_log_cols = np.log(1 / (1 + X_test[:, inv_log_cols]))
    X_test = np.hstack((X_test, X_test_inv_log_cols))

    # Scaling the features.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X, X_test


#########################################################################
# Training run
#########################################################################

def train_and_predict(X, W, Y, X_test):
    # Preprocess the features.
    X, X_test = preprocess(X, X_test)

    classifier = nested_model()

    # Train the model.
    classifier.fit(X, Y, sample_weight=W)

    Y_pred_proba = classifier.predict_proba(X)
    Y_test_pred_proba = classifier.predict_proba(X_test)

    # Thresholding for prediction.
    signal_threshold = 83
    cut = np.percentile(Y_test_pred_proba[:, 1], signal_threshold)
    thresholded_Y_pred = Y_pred_proba[:, 1] > cut
    thresholded_Y_test_pred = Y_test_pred_proba[:, 1] > cut

    return [Y_test_pred_proba[:, 1], thresholded_Y_test_pred]


#########################################################################
# Submission generation
#########################################################################

def write_submission_file(ids_test, Y_test_pred, thresholded_Y_test_pred):
    ids_probs = np.transpose(np.vstack((ids_test, Y_test_pred)))
    ids_probs = np.array(sorted(ids_probs, key=lambda x: -x[1]))
    ids_probs_ranks = np.hstack((
        ids_probs,
        np.arange(1, ids_probs.shape[0] + 1).reshape((ids_probs.shape[0], 1))))

    test_ids_map = {}
    for test_id, prob, rank in ids_probs_ranks:
        test_id = int(test_id)
        rank = int(rank)
        test_ids_map[test_id] = rank

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EventId", "RankOrder", "Class"])
        for i, pred in enumerate(thresholded_Y_test_pred):
            event_id = int(ids_test[i])
            rank = test_ids_map[ids_test[i]]
            klass = "s" if pred else "b"
            writer.writerow([event_id, rank, klass])


#########################################################################
# Data loading and execution
#########################################################################

# Fix CPU affinity caused by Numpy.
os.system("taskset -p 0xffffffff %d" % os.getpid())

# It is important to pick the right seed! Reduce randomness wherever
# possible. Especially in a CV loop, so your solutions are more
# comparable.
seed = 512
random.seed(seed)

# Path to the zip files
training_zip_path = "training.zip"
test_zip_path = "test.zip"

# Load training data from zip
with zipfile.ZipFile(FILE_PATH + training_zip_path, "r") as z:
    with z.open("training.csv") as f:
        df_train = pd.read_csv(f)

# Load test data from zip
with zipfile.ZipFile(FILE_PATH + test_zip_path, "r") as z:
    with z.open("test.csv") as f:
        df_test = pd.read_csv(f)

# Prepare training data
X = df_train.iloc[:, 1:31].values
Y = df_train["Label"].apply(lambda x: 1 if x == "s" else 0).values
W = df_train["Weight"].values

# Prepare test data
ids_test = df_test["EventId"].values
X_test = df_test.iloc[:, 1:31].values

X, X_test = preprocess(X, X_test)

classifier = nested_model()
