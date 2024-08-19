import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

train_data = pd.read_csv('workspace/hyperopt/higgs-boson/data/training.zip')
test_data = pd.read_csv('workspace/hyperopt/higgs-boson/data/test.zip')


# DELETED WEIGHT BEACAUSE IT WAS NOT IN THE TESTING SET SO I DON'T WANT TO TRAIN THE MODEL ON THE TRAINING SET WHICH HAS
# AN EXTRA VARIABLE WHICH IS TO BE CONSIDERED BY THE ALGORITHMS
train_data = train_data.drop(['Weight'], axis=1)

# GETTING THE DUMMY OF THE LABEL
dummy = pd.DataFrame()
dummy = pd.get_dummies(train_data['Label'], drop_first=True)

train_data = pd.concat([train_data, dummy], axis=1)
train_data = train_data.drop(['Label'], axis=1)

x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
skf = StratifiedKFold(n_splits=5)
scores = []
for x_fold_idx, y_fold_idx in skf.split(x, y):
    x_fold = x.iloc[x_fold_idx]
    y_fold = y.iloc[y_fold_idx]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)

    # DATA PREPROCESSING IS ALWAYS NECESSARY SO THAT THE MODELS WOULD NOT HAVE TO WORRY ABT ANYTHING
    # WE MAY NOT ALSO RUN INTO ANY KERNEL RELATED ISSUES
    # TIMING PROBLEM WOULDN'T HAPPEN( IT HAPPENED WITH ME SO AN ADVICE)
    # ALWAYS PREPROCESS (I USED STANDARD SCALER), YOU CAN ALSO NORMALISE IT
    eventid_test = test_data.copy()
    sc = StandardScaler()
    test_data = sc.fit_transform(test_data)

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    xg = XGBClassifier()
    xg.fit(x_train, y_train)

    cb = CatBoostClassifier()

    xg_pred = xg.predict(x_test)
    print(classification_report(y_test, xg_pred))

    xg.score(x_train, y_train)
    xg.score(x_test, y_test)
    scores.append(1 - accuracy_score(y_test, xg_pred))

score = np.mean(scores)