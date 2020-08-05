import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\business\hotelsevolclean_comma.csv")

print(df.columns)
print("======")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
print("======")
print(df.info())
print("======")
print(df.isnull().any())

df = df.loc[(df['Classement'] == 'All') & (df['Quartier'] == 'All')]


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values(by=['Date'], inplace=True)
df = df.set_index(pd.DatetimeIndex(df['Date']))
index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')

df = df.drop(['Source', 'Anne', 'Quartier', 'Classement', 'Date'], 1)

df.plot(y='Average Price')
plt.show()

corr = df.corr()
plt.subplots(figsize=(8,8))
sns.heatmap(corr, cmap="Blues", cbar=False, annot=True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
from timeit import default_timer as timer

model = VAR(df)
results = model.fit()
print(results.summary())
model_fit = model.fit()
forecast = model_fit.forecast(model_fit.y, steps=6)
print(forecast)