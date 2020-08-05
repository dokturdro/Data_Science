import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from random import *

df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\medical\chronicdiseases_brazil.csv")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
print(df.info())
print(df.isnull().any())


df.rename(columns={"No-show": "noshow"}, inplace=True)
gender = pd.get_dummies(df.Gender).iloc[:,1:]
noshow = pd.get_dummies(df.noshow).iloc[:,1:]
df = pd.concat([df, gender, noshow], 1)
df = df.drop(['PatientId', 'AppointmentDay', 'AppointmentID', 'ScheduledDay', 'Neighbourhood', 'Yes', 'Gender', 'noshow'],1)

print(df.info())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

X = df.drop('M', 1)
y = df['M']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
# param_dist = {'n_estimators': randint(150, 500),
#               'learning_rate': uniform(0.01, 0.07),
#               'subsample': uniform(0.3, 0.7),
#               'max_depth': randint(3, 9),
#               'colsample_bytree': uniform(0.5, 0.45),
#               'min_child_weight': randint(1, 2, 3)
#              }
# clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 25, scoring = 'f1', error_score = 0, verbose = 3, n_jobs = -1)

# numFolds = 5
# folds = cross_validation.KFold(n = len(X), shuffle = True, n_folds = numFolds)

# estimators = []
# results = np.zeros(len(X))
# score = 0.0
# for train_index, test_index in folds:
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(X_train, y_train)

#     estimators.append(clf.best_estimator_)
#     results[test_index] = clf.predict(X_test)
#     score += f1_score(y_test, results[test_index])
# score /= numFolds

classifier = XGBClassifier( 
             silent=0,
             learning_rate=0.1,
             n_estimators=100,
             max_depth=10,
             min_child_weight=1,
             gamma=0,
             subsample=0.5,
             colsample_bytree=0.5,
             objective='reg:logistic',
             nthread=4,
             scale_pos_weight=1,
             seed=27
             )

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(classifier, X, y, cv=kfold)
print("Accuracy K-fold: %.2f (%.2f)" % (results.mean(), results.std()))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : %.2f" % (accuracy))

# eval_set = [(X_test, y_test)]
# model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)