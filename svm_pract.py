import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# load dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\worldstat\avocado.csv")

# basic exploration for shape and NaNs
print(df.columns)
print("======")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
print("======")
print(df.info())
print("======")
print(df.isnull().any())

df = df.drop(df.columns[:2],1)
print(df.head())


# checkout autocorrelation matrix for redundant data
corr = df.corr()
sns.heatmap(corr, annot= True)
plt.show()

# replace column name so it doesn't get overriden by Python
df.columns = df.columns.str.replace('type','label')

# setup dummy variables for label columns
label = pd.get_dummies(df.label).iloc[:,1:]
year = pd.get_dummies(df.year).iloc[:,1:]
region = pd.get_dummies(df.region).iloc[:,1:]
df = pd.concat([df, label, year, region], 1)
df = df.drop(['label', 'year', 'region'],1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNetCV

# setup label
X = df.drop('organic', 1)
y = df['organic']

# split data for the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# classifiers to run and check
classifiers = [['DecTree :',DecisionTreeRegressor()],
               ['RandFor :',RandomForestRegressor()],
               ['KNeighb :', KNeighborsRegressor(n_neighbors = 5)],
               ['SVRegre :', SVR()],
               ['SVClass :', SVC()],
               ['GBClass :', GradientBoostingRegressor()],
               ['ExTRegr :', ExtraTreesRegressor(n_estimators=1, min_samples_split=5)],
               ['ElNetCV :', ElasticNetCV()]]

# print out benchmark functions for classifiers
print("\n====== RMSE ======")
for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))

print("\n====== R^2 ======")
for name,classifier in classifiers:
    print(name, (classifier.score(X_test, y_test)))

# fit a selected classifier
classifier = RandomForestRegressor(n_estimators=1)
classifier.fit(X_train, y_train)

# run prediction..
y_pred = classifier.predict(X_test)

# ..and check it out on an interpreter matrix instead of seaborn
print("\n====== Confusion Matrix ======")
##print(confusion_matrix(y_test,y_pred))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
