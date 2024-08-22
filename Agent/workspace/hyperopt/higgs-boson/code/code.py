import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# FILE_PATH= "../data/"
FILE_PATH= "./workspace/hyperopt/higgs-boson/data/"

TARGET = "NObeyesdad"
submission_path="ori_submission.csv"
n_splits = 9
RANDOM_SEED = 73

train_data = pd.read_csv(FILE_PATH+'training.zip')
test_data = pd.read_csv(FILE_PATH+'test.zip')

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

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
# xg.fit(x_train, y_train)

# cb = CatBoostClassifier()

# xg_pred = xg.predict(x_test)
# print(classification_report(y_test, xg_pred))

# xg.score(x_train, y_train)
# xg.score(x_test, y_test)

# final_pred = xg.predict(test_data)
# final_pred = pd.DataFrame(final_pred)
# final_pred.replace(to_replace = (0, 1), value = ('b', 's'), inplace = True)
# final_pred.rename(columns = {0: 'Class'}, inplace = True)
# final_pred['RankOrder'] = final_pred['Class'].argsort().argsort() + 1  
# pred = eventid_test['EventId'] #this is why I created a copy in cell 24

# pred = pd.DataFrame(pred)


# final_pred = final_pred[['RankOrder', 'Class']]
# result = [pred, final_pred]


# final_pred = pd.concat(result, axis =1)
# final_pred.to_csv(submission_path, index = False)
# score = 1 - accuracy_score(y_test, xg_pred)