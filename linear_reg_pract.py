import pandas as pd
import quandl, math
import numpy as np
import datetime
from scipy import sparse
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "EzAELjGNysBk6Nmf5Zn4"

df = quandl.get("WSE/PLAY")

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['hl_pct'] = (df['High'] - df['Close']) / df['Close'] * 100
df['pct_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
df = df[['Close', 'hl_pct', 'pct_change', 'Volume']]

forecast_col = 'Close'
df.fillna(1, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.fillna(1, inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df)

X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

print(len(X), len(y))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

clf = LinearRegression()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, acc, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] =[np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()