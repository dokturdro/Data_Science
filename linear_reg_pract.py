import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Administrator\Desktop\datasets\crypto\crypto-markets.csv')
df = df.loc[df['name'] == 'Litecoin']

print(df.columns)
print("======")
print(df.head(20))
print("======")
print(df.info())
print("======")
print(df.isnull().any())

df = df.drop(['symbol', 'slug', 'ranknow', 'spread', 'close_ratio'], 1)
print(df.columns)
print(df.describe())

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(df.head())


##df.close.resample('W').mean().plot()
##plt.show()
##df["2017-06-01":].close.plot()
##plt.show()

df = df["2017-06-01":]
df['hilo'] = (df['high'] - df['close']) / df['close'] * 100
df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100
print(df.corr()["close"])
df = df.drop('name', 1)

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler

X = df.drop('close', 1)
y = df['close']

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

linreg_r2 = model.score(X_test, y_test)
print(linreg_r2)

predict = model.predict(X_train)

##print(predict)

plt.plot(df['close'], color = 'blue', label = "Litecoin price")
##plt.plot(predict, color = 'g', label = "Predicted crypto price")
plt.title('Crypto price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()