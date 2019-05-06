import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset with singling out the rows with desired crypto
df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\business\939775908_T_ONTIME_REPORTING.csv")

# basic exploration for shape and NaNs
print(df.columns)
print("======")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

df = df.drop(['DEP_DEL15', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF',
              'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'DIVERTED', 
              'CRS_DEP_TIME', 'DEP_TIME', 'DIV_AIRPORT_LANDINGS'], 1)
print("======")
print(df.info())
print("======")
print(df.isnull().any())
print("======")
print(df.head())
# setting up date column as the index for time series
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df = df.set_index('FL_DATE')
df = df.sort_values(by='FL_DATE')
df = df.fillna(0)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

corr = df.corr()
plt.show()

YEAR = pd.get_dummies(df.YEAR).iloc[:,1:]
QUARTER = pd.get_dummies(df.QUARTER).iloc[:,1:]
MONTH = pd.get_dummies(df.MONTH).iloc[:,1:]
DAY_OF_MONTH = pd.get_dummies(df.DAY_OF_MONTH).iloc[:,1:]
CANCELLATION_CODE = pd.get_dummies(df.CANCELLATION_CODE).iloc[:,1:]
OP_UNIQUE_CARRIER = pd.get_dummies(df.OP_UNIQUE_CARRIER).iloc[:,1:]
ORIGIN = pd.get_dummies(df.ORIGIN).iloc[:,1:]
DEST = pd.get_dummies(df.DEST).iloc[:,1:]
df = pd.concat([df, OP_UNIQUE_CARRIER, ORIGIN, DEST, CANCELLATION_CODE], 1)
df = df.drop(['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CANCELLATION_CODE'],1)
print(df.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier
from timeit import default_timer as timer

X = df.drop('CANCELLED', 1)
y = df['CANCELLED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# classifiers to run and check
regressors = [['DecTree :', DecisionTreeRegressor()],
               ['RandFor :', RandomForestRegressor()],
               ['ExTRegr :', ExtraTreesRegressor(n_estimators=1, min_samples_split=5)]]

# print out benchmark functions for classifiers
print("\n====== RMSE ======")
for name,regressor in regressors:
    start = timer()
    regressor = regressor
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    end = timer()
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
    print(end - start)

print("\n====== R^2 ======")
for name,regressor in regressors:
    print(name, (regressor.score(X_test, y_test)))

# fit a selected classifier
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# run prediction..
y_pred = regressor.predict(X_test)

importances = regressor.feature_importances_
#Sort it
print("Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print(sorted_feature_importance)
# and plot it in seaborn
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
cm_df = pd.DataFrame(cm,
                     index = ['1', '0'], 
                     columns = ['1', '0'])
ax = sns.heatmap(cm_df, fmt='d', cmap="Blues", cbar=False,  annot=True)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
