import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv(r'C:\Users\Administrator\Desktop\datasets\crypto\crypto-markets.csv')
df = df.loc[df['name'] == 'Litecoin']

print(df.columns)
print(df.head(20))
print(df.info())
print(df.isnull().any())

df = df.drop(['symbol', 'slug', 'ranknow', 'spread', 'close_ratio'], 1)
print(df.columns)
print(df.describe())

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(df.head())

df = df["2017-06-01":]
df['hilo'] = (df['high'] - df['close']) / df['close'] * 100
df['pct_change'] = (df['close'] - df['open']) / df['open'] * 100
print(df.corr()["close"])
df = df.drop('name', 1)

print('\n')
print(df.head())
print('\n')


from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

##X = df.drop('close', 1)
##y = df['close']

forecast_col = 'close'
forecast_out = int(60)

df['label'] = df[forecast_col].shift(-forecast_out)
print('\n')
print(df.head())
print('\n')

X = np.array(df.drop(['label'], 1))
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
X_forecast_out = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['label'])
y = y[:-forecast_out]

##scaler = MinMaxScaler(feature_range=(0,1))
##X = scaler.fit_transform(X)
##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifiers = [['Lasso: ', Lasso()],
               ['Ridge: ', Ridge()],
               ['LinearRegression: ', LinearRegression()]]

print("====== RMSE ======")
for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))

print("====== Accuracy score ======")
for name,classifier in classifiers:
    print(name, (classifier.score(X_test, y_test)))

model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)

print(predict)

##last_close = df['date'][-1]
##last_date = df.iloc[-1].name.timestamp()
##
##for i in range(90):
##    last_close *= modifier
##    next_date = datetime.datetime.fromtimestamp(last_date)
##    last_date += 86400
##
##    # Outputs data into DataFrame to enable plotting
##    df.loc[next_date] = [np.nan, last_close]
##
##plt.plot(df['close'], color = 'blue', label = "Litecoin price")
##plt.plot(predict, color = 'g', label = "Predicted crypto price")
##plt.title('Crypto price prediction')
##plt.xlabel('Time')
##plt.ylabel('Price')
##plt.legend()
##plt.show()