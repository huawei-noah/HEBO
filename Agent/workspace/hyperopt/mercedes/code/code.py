#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here"s several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

FILE_PATH = "../data/"
# FILE_PATH= "./workspace/hyperopt/mercedes/data/"
TARGET = "NObeyesdad"
submission_path = "ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

# readin csv file
df = pd.read_csv(FILE_PATH + "train.csv")

# Here I have droped X0,X5 because some unique values were different in test data
# I checked it didn"t affected score much
df.drop(["ID", "X0", "X5"], axis=1, inplace=True)


# features(x1,X2,X6,X8) have large no. of unique values
# this function top 10 most occuring unique values amongst others
def onehot(data, variable):
    t1 = [x for x in df[variable].value_counts().sort_values(ascending=False).head(10).index]
    for i in t1:
        df[variable + "" + i] = np.where(df[variable] == i, 1, 0)


onehot(df, "X1")
onehot(df, "X2")
onehot(df, "X6")
onehot(df, "X8")

# droping down columns after getting dummies 
dum = pd.get_dummies(df[["X3", "X4"]], drop_first=True)
df = pd.concat([dum, df], axis=1)
df.drop(["X1", "X2", "X3", "X4", "X6", "X8"], axis=1, inplace=True)

# reading test data
dft = pd.read_csv(FILE_PATH + "test.csv")

# Here I have droped X0,X5 because some unique values were different in train data
# I checked it didn"t affected score much
dft.drop(["ID", "X0", "X5"], axis=1, inplace=True)


# features(x1,X2,X6,X8) have large no. of unique values
# this function top 10 most occuring unique values amongst others
def onehot(data, variable):
    t1 = [x for x in dft[variable].value_counts().sort_values(ascending=False).head(10).index]
    for i in t1:
        dft[variable + "" + i] = np.where(dft[variable] == i, 1, 0)


onehot(dft, "X1")
onehot(dft, "X2")
onehot(dft, "X6")
onehot(dft, "X8")

# droping down columns after getting dummies 
dum = pd.get_dummies(dft[["X3", "X4"]], drop_first=True)
dft = pd.concat([dum, dft], axis=1)
dft.drop(["X1", "X2", "X3", "X4", "X6", "X8"], axis=1, inplace=True)

# defining x and y
x = df.drop(["y"], axis=1)
y = df["y"]
y = y.astype("int")

# This for feature selection 
# Selecting top 15 features
feat = SelectKBest(score_func=chi2, k="all")
fit = feat.fit(x, y)
dfscore = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
best = pd.concat([dfscore, dfcolumns], axis=1)
best.columns = ["score", "specs"]
d = print(best.nlargest(15, "score"))

# they  were not present train data and test data
df.drop("X6b", axis=1, inplace=True)
dft.drop("X6f", axis=1, inplace=True)

# setting order for both data
dft = dft[
    ["X3_b", "X3_c", "X3_d", "X3_e", "X3_f", "X3_g", "X4_b", "X4_c", "X4_d", "X10", "X11", "X12", "X13", "X14", "X15",
     "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X26", "X27", "X28", "X29", "X30", "X31", "X32",
     "X33", "X34", "X35", "X36", "X37", "X38", "X39", "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48",
     "X49", "X50", "X51", "X52", "X53", "X54", "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64",
     "X65", "X66", "X67", "X68", "X69", "X70", "X71", "X73", "X74", "X75", "X76", "X77", "X78", "X79", "X80", "X81",
     "X82", "X83", "X84", "X85", "X86", "X87", "X88", "X89", "X90", "X91", "X92", "X93", "X94", "X95", "X96", "X97",
     "X98", "X99", "X100", "X101", "X102", "X103", "X104", "X105", "X106", "X107", "X108", "X109", "X110", "X111",
     "X112", "X113", "X114", "X115", "X116", "X117", "X118", "X119", "X120", "X122", "X123", "X124", "X125", "X126",
     "X127", "X128", "X129", "X130", "X131", "X132", "X133", "X134", "X135", "X136", "X137", "X138", "X139", "X140",
     "X141", "X142", "X143", "X144", "X145", "X146", "X147", "X148", "X150", "X151", "X152", "X153", "X154", "X155",
     "X156", "X157", "X158", "X159", "X160", "X161", "X162", "X163", "X164", "X165", "X166", "X167", "X168", "X169",
     "X170", "X171", "X172", "X173", "X174", "X175", "X176", "X177", "X178", "X179", "X180", "X181", "X182", "X183",
     "X184", "X185", "X186", "X187", "X189", "X190", "X191", "X192", "X194", "X195", "X196", "X197", "X198", "X199",
     "X200", "X201", "X202", "X203", "X204", "X205", "X206", "X207", "X208", "X209", "X210", "X211", "X212", "X213",
     "X214", "X215", "X216", "X217", "X218", "X219", "X220", "X221", "X222", "X223", "X224", "X225", "X226", "X227",
     "X228", "X229", "X230", "X231", "X232", "X233", "X234", "X235", "X236", "X237", "X238", "X239", "X240", "X241",
     "X242", "X243", "X244", "X245", "X246", "X247", "X248", "X249", "X250", "X251", "X252", "X253", "X254", "X255",
     "X256", "X257", "X258", "X259", "X260", "X261", "X262", "X263", "X264", "X265", "X266", "X267", "X268", "X269",
     "X270", "X271", "X272", "X273", "X274", "X275", "X276", "X277", "X278", "X279", "X280", "X281", "X282", "X283",
     "X284", "X285", "X286", "X287", "X288", "X289", "X290", "X291", "X292", "X293", "X294", "X295", "X296", "X297",
     "X298", "X299", "X300", "X301", "X302", "X304", "X305", "X306", "X307", "X308", "X309", "X310", "X311", "X312",
     "X313", "X314", "X315", "X316", "X317", "X318", "X319", "X320", "X321", "X322", "X323", "X324", "X325", "X326",
     "X327", "X328", "X329", "X330", "X331", "X332", "X333", "X334", "X335", "X336", "X337", "X338", "X339", "X340",
     "X341", "X342", "X343", "X344", "X345", "X346", "X347", "X348", "X349", "X350", "X351", "X352", "X353", "X354",
     "X355", "X356", "X357", "X358", "X359", "X360", "X361", "X362", "X363", "X364", "X365", "X366", "X367", "X368",
     "X369", "X370", "X371", "X372", "X373", "X374", "X375", "X376", "X377", "X378", "X379", "X380", "X382", "X383",
     "X384", "X385", "X1aa", "X1s", "X1b", "X1l", "X1v", "X1r", "X1i", "X1a", "X1c", "X1o", "X2as", "X2ae", "X2ai",
     "X2m", "X2ak", "X2r", "X2n", "X2s", "X2f", "X2e", "X6g", "X6j", "X6d", "X6i", "X6l", "X6a", "X6h", "X6k", "X6c",
     "X8j", "X8s", "X8f", "X8n", "X8i", "X8e", "X8r", "X8a", "X8w", "X8v"]]
df = df[
    ["X3_b", "X3_c", "X3_d", "X3_e", "X3_f", "X3_g", "X4_b", "X4_c", "X4_d", "X10", "X11", "X12", "X13", "X14", "X15",
     "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X26", "X27", "X28", "X29", "X30", "X31", "X32",
     "X33", "X34", "X35", "X36", "X37", "X38", "X39", "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48",
     "X49", "X50", "X51", "X52", "X53", "X54", "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64",
     "X65", "X66", "X67", "X68", "X69", "X70", "X71", "X73", "X74", "X75", "X76", "X77", "X78", "X79", "X80", "X81",
     "X82", "X83", "X84", "X85", "X86", "X87", "X88", "X89", "X90", "X91", "X92", "X93", "X94", "X95", "X96", "X97",
     "X98", "X99", "X100", "X101", "X102", "X103", "X104", "X105", "X106", "X107", "X108", "X109", "X110", "X111",
     "X112", "X113", "X114", "X115", "X116", "X117", "X118", "X119", "X120", "X122", "X123", "X124", "X125", "X126",
     "X127", "X128", "X129", "X130", "X131", "X132", "X133", "X134", "X135", "X136", "X137", "X138", "X139", "X140",
     "X141", "X142", "X143", "X144", "X145", "X146", "X147", "X148", "X150", "X151", "X152", "X153", "X154", "X155",
     "X156", "X157", "X158", "X159", "X160", "X161", "X162", "X163", "X164", "X165", "X166", "X167", "X168", "X169",
     "X170", "X171", "X172", "X173", "X174", "X175", "X176", "X177", "X178", "X179", "X180", "X181", "X182", "X183",
     "X184", "X185", "X186", "X187", "X189", "X190", "X191", "X192", "X194", "X195", "X196", "X197", "X198", "X199",
     "X200", "X201", "X202", "X203", "X204", "X205", "X206", "X207", "X208", "X209", "X210", "X211", "X212", "X213",
     "X214", "X215", "X216", "X217", "X218", "X219", "X220", "X221", "X222", "X223", "X224", "X225", "X226", "X227",
     "X228", "X229", "X230", "X231", "X232", "X233", "X234", "X235", "X236", "X237", "X238", "X239", "X240", "X241",
     "X242", "X243", "X244", "X245", "X246", "X247", "X248", "X249", "X250", "X251", "X252", "X253", "X254", "X255",
     "X256", "X257", "X258", "X259", "X260", "X261", "X262", "X263", "X264", "X265", "X266", "X267", "X268", "X269",
     "X270", "X271", "X272", "X273", "X274", "X275", "X276", "X277", "X278", "X279", "X280", "X281", "X282", "X283",
     "X284", "X285", "X286", "X287", "X288", "X289", "X290", "X291", "X292", "X293", "X294", "X295", "X296", "X297",
     "X298", "X299", "X300", "X301", "X302", "X304", "X305", "X306", "X307", "X308", "X309", "X310", "X311", "X312",
     "X313", "X314", "X315", "X316", "X317", "X318", "X319", "X320", "X321", "X322", "X323", "X324", "X325", "X326",
     "X327", "X328", "X329", "X330", "X331", "X332", "X333", "X334", "X335", "X336", "X337", "X338", "X339", "X340",
     "X341", "X342", "X343", "X344", "X345", "X346", "X347", "X348", "X349", "X350", "X351", "X352", "X353", "X354",
     "X355", "X356", "X357", "X358", "X359", "X360", "X361", "X362", "X363", "X364", "X365", "X366", "X367", "X368",
     "X369", "X370", "X371", "X372", "X373", "X374", "X375", "X376", "X377", "X378", "X379", "X380", "X382", "X383",
     "X384", "X385", "X1aa", "X1s", "X1b", "X1l", "X1v", "X1r", "X1i", "X1a", "X1c", "X1o", "X2as", "X2ae", "X2ai",
     "X2m", "X2ak", "X2r", "X2n", "X2s", "X2f", "X2e", "X6g", "X6j", "X6d", "X6i", "X6l", "X6a", "X6h", "X6k", "X6c",
     "X8j", "X8s", "X8f", "X8n", "X8i", "X8e", "X8r", "X8a", "X8w", "X8v", "y"]]

x = df.drop("y", axis=1)
y = df["y"]

xg = xgboost.XGBRegressor(verbosity=0, device="gpu")

# xg.fit(x,y)


# pred = xg.predict(dft)


# # xgboost gives the best score among others
# # Using it as final algorithm for prediction


# # reading submission file and concating pred value and ID column 
# t = pd.read_csv(FILE_PATH+"sample_submission.csv")
# ypred = pd.DataFrame(pred)
# sub = pd.concat([t["ID"],ypred],axis=1)


# sub.columns = ["ID","y"]
# sub.to_csv(submission_path,index=False)

# # checking for cross validation score
# score = cross_val_score(xg,x,y,cv=5).mean()
