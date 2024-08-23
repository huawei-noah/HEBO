import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def k_folds_cv(model, X: pd.DataFrame, y: pd.DataFrame, metric_func):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    try:
        for fold_i, (train_index, valid_index) in enumerate(strat_cv.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            score = metric_func(y_valid, y_pred)
            scores.append(score)
            # print(f"FOLD {fold_i} Done. Score : {score}")

    except ValueError:
        for fold_i, (train_index, valid_index) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            score = metric_func(y_valid, y_pred)
            scores.append(score)

    mean_score = np.mean(scores)
    return mean_score
