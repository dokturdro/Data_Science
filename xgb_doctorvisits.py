import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error


df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\medical\chronicdiseases_brazil.csv")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
print(df.info())
print(df.isnull().any())


df.rename(columns={"No-show": "noshow"}, inplace=True)
nbhood = pd.get_dummies(df.Neighbourhood).iloc[:,1:]
gender = pd.get_dummies(df.Gender).iloc[:,1:]
noshow = pd.get_dummies(df.noshow).iloc[:,1:]
df = pd.concat([df, nbhood, gender, noshow], 1)
df = df.drop(['AppointmentDay', 'ScheduledDay', 'Neighbourhood', 'Yes', 'Gender'],1)

print(df.info())

X = df.drop('noshow', 1)
y = df['noshow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = XGBClassifier(
             silent=0,
             learning_rate=0.1,
             n_estimators=100,
             max_depth=5,
             min_child_weight=1,
             gamma=0,
             subsample=0.8,
             colsample_bytree=0.8,
             objective='binary:logistic',
             nthread=4,
             scale_pos_weight=1,
             seed=27
             )

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
