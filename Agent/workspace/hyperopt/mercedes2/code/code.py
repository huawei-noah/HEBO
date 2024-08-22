# %% [code]
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb

# FILE_PATH = "../data/"
FILE_PATH = "./workspace/hyperopt/mercedes2/data/"
TARGET = "NObeyesdad"
submission_path = "ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

train = pd.read_csv(FILE_PATH + "train.csv")
test = pd.read_csv(FILE_PATH + "test.csv")

y_train = train["y"].values
y_mean = np.mean(y_train)
id_test = test["ID"]

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(["ID", "y"], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]


class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))


#
# Model/pipeline with scaling,pca,svm
#
svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            PCA(),
                                            SVR(kernel="rbf", C=1.0, epsilon=0.05)]))

# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring="r2")
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()

#
# Model/pipeline with scaling,pca,ElasticNet
#
en_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           PCA(n_components=125),
                                           ElasticNet(alpha=0.001, l1_ratio=0.1)]))

#
# XGBoost model
#
xgb_model = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.921,
                                     objective="reg:linear", n_estimators=1300, base_score=y_mean)

xgb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=10)),
                                      AddColumns(transform_=FastICA(n_components=10, max_iter=500)),
                                      xgb_model]))

# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring="r2")
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#
rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                 min_samples_leaf=25, max_depth=3)


# results = cross_val_score(rf_model, train, y_train, cv=5, scoring="r2")
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Now the training and stacking part.  In previous version i just tried to train each model and
# find the best combination, that lead to a horrible score (Overfit?).  Code below does out-of-fold
# training/predictions and then we combine the final results.
#
# Read here for more explanation (This code was borrowed/adapted) :
#

class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def init_stacker(self):
        return self.stacker


stack = Ensemble(n_splits=5,
                 # stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker=ElasticNet(l1_ratio=0.1, alpha=1.4),
                 base_models=(svm_pipe, en_pipe, xgb_pipe, rf_model))
stacker = stack.init_stacker()
