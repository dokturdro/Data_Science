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

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
print(df.head())

df = df.loc[df['region'] == 'TotalUS']
df.plot(y='AveragePrice')
plt.show()

# checkout autocorrelation matrix for redundant data
corr = df.corr()
plt.subplots(figsize=(8,8))
sns.heatmap(corr, cmap="Blues", cbar=False, annot=True)
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from timeit import default_timer as timer

model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())