import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv(r'C:\Users\Administrator\Desktop\datasets\crypto\crypto-markets.csv')
df = df.loc[df['name'] == 'Bitcoin']

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

df = df["2017-01-01":]
df['oc_diff'] = df['close']- df['open']
df['hilo'] = (df['high'] - df['close']) / df['close'] * 100
df['daily_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
##df['mov_avg'] = df.rolling(['close'], window=5).mean()
print(df.corr()["daily_avg"])
df = df.drop(['name', 'volume'], 1)

print('\n')
print(df.head())
print('\n')

from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df['forecast'] = df['daily_avg'].shift(-90)
X = df.dropna().drop(['forecast'], axis=1)
y = df.dropna()['forecast']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
forecast =  df.tail(90).drop(['forecast'], 1)

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)


classifiers = [['Lasso: ', Lasso()],
               ['Ridge: ', Ridge()],
               ['LinearRegression: ', LinearRegression()],
               ['Random Forest Regressor: ', RandomForestRegressor(n_estimators=200)],
               ['Bayesian Ridge: ', BayesianRidge()],
               ['Elastic Net CV: ', ElasticNetCV()]]

print("====== RMSE ======")
for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))

print("====== Accuracy score ======")
for name,classifier in classifiers:
    print(name, (classifier.score(X_test, y_test)))


model = RandomForestRegressor()
model.fit(X_train, y_train)

predict = model.predict(forecast)

print(predict)

plt.figure(figsize=(15,8))
(df[:-90]['daily_avg']).plot(label='Historical Price')
(df[-91:]['daily_avg']).plot(label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price in USD')
plt.title('Prediction on Daily Average Price of Bitcoin')
plt.legend()
plt.show()



##last_close = df.iloc[:,0][-1]
##last_date = df.iloc[-1].name.timestamp()
##
##for i in range(30):
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