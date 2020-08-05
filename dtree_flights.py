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

df = df.drop(['ARR_DEL15', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF', 'ORIGIN',
              'WHEELS_ON', 'YEAR', 'TAXI_IN', 'TAXI_OUT', 'CRS_ARR_TIME', 'ARR_TIME', 'DIVERTED', 
              'CRS_DEP_TIME', 'QUARTER', 'DEP_TIME', 'DIV_AIRPORT_LANDINGS', 'DEST', 
              'OP_UNIQUE_CARRIER', 'CANCELLATION_CODE'], 1)
print("======")
print(df.info())
print("======")
print(df.isnull().any())
print("======")
print(df.columns)

# setting up date column as the index for time series
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df = df.set_index('FL_DATE')
df = df.sort_values(by='FL_DATE')
df = df.dropna(how='all', axis='columns')
df = df.fillna(0)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())

corr = df.corr()
plt.subplots(figsize=(6,6))
sns.heatmap(corr, annot=True)
# plt.show()

print(df['CANCELLED'])

# CANCELLATION_CODE = pd.get_dummies(df.CANCELLATION_CODE).iloc[:,1:]
# OP_UNIQUE_CARRIER = pd.get_dummies(df.OP_UNIQUE_CARRIER).iloc[:,1:]
# ORIGIN = pd.get_dummies(df.ORIGIN).iloc[:,1:]
# df = pd.concat([df, OP_UNIQUE_CARRIER, ORIGIN, CANCELLATION_CODE], 1)
# df = df.drop(['OP_UNIQUE_CARRIER', 'ORIGIN', 'CANCELLATION_CODE'],1)

print(df.shape)

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
              ['RandFor :', RandomForestRegressor()]]

# print out benchmark functions for classifiers
print("\n====== RMSE ======")
for name,regressor in regressors:
    start = timer()
    regressor = regressor
    regressor.fit(X_train, y_train)
    prediction = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    end = timer()
    time = end - start
    print(name, "% .3f" % rmse)
    print("% .3f" % time,"s")

print("\n====== R^2 ======")
r2_result = []
for name,regressor in regressors:
    score = regressor.score(X_test, y_test)
    r2_result.append([regressor, score])
    print(name, "% .3f" % score)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

importances = regressor.feature_importances_
feature_importances = pd.DataFrame(regressor.feature_importances_,
                                   index = df.columns[:14],
                                   columns=['importance']).sort_values('importance',                     
                                   ascending=False)

print("====== Importance ======")
print(feature_importances)

print("============")
print(y_test)
print(y_pred)

cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
cm_df = pd.DataFrame(cm,
                     index = ['1', '0'], 
                     columns = ['1', '0'])
ax = sns.heatmap(cm_df, fmt='d', cmap="Blues", cbar=False, annot=True)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()
