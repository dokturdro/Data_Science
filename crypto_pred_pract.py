import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset with singling out the rows with desired crypto
df = pd.read_csv(r'C:\Users\Administrator\Desktop\datasets\crypto\crypto-markets.csv')
df = df.loc[df['name'] == 'Bitcoin']

# basic exploration for data organization and possible noise
print(df.columns)
print(df.head(20))
print(df.info())
print(df.isnull().any())

# dropping columns that will have little or counterproductive effect on the process
df = df.drop(['symbol', 'slug', 'ranknow', 'spread', 'close_ratio'], 1)
print(df.columns)
print(df.describe())

# setting up date column as the index for time series
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(df.head())

# selecting the beginning date of relevant data, adding some features, dropping other
df = df['2017-01-01':]
df['daily_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
df['hilo'] = (df['high'] - df['close']) / df['close'] * 100

# adding rolling average feature, taken from previous 6 entries, filling in NaN
df_rolling_avg = df['close']
df_rolling_avg = df_rolling_avg.rolling(window=6).mean()
df_rolling_avg = df_rolling_avg.rename('rolling_avg', inplace=True)
df = pd.concat([df, df_rolling_avg],1)
df = df.fillna(method='backfill')

# check for autocorrelation
corr = df.corr()
plt.subplots(figsize=(6,6))
sns.heatmap(corr, annot=True)
plt.show()


df = df.drop(['name', 'volume'], 1)

# importing sklearn ML modules
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNetCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# shifting data to make space for a 3 months worth forecast, splitting for training
df['forecast'] = df['daily_avg'].shift(-90)
X = df.dropna().drop(['forecast'], axis=1)
y = df.dropna()['forecast']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
forecast =  df.tail(90).drop(['forecast'], 1)

# rescaling data
def rescale(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    return X
# list of lists with models to be tested
classifiers = [['LinReg: ', LinearRegression()],
               ['RFrReg: ', RandomForestRegressor(n_estimators=100)],
               ['BayesR: ', BayesianRidge()],
               ['ExTReg: ', ExtraTreesRegressor(n_estimators=200, min_samples_split=5)],
               ['ENetCV: ', ElasticNetCV()]]

# printing tested models' accuracy with root mean squared error and r squared
print("====== RMSE ======")
for name, classifier in classifiers:
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    print(name, "% .2f" % rmse)

print("====== R^2 ======")
r2_result = []
for name, classifier in classifiers:
    r2_result.append([classifier, classifier.score(X_test, y_test)])
    # top_regressor = max(r2_result, key=r2_result.get)
print(r2_result)
r2_result.sort(key=lambda x: x[1], reverse=True)
model = r2_result[0][0]

print(r2_result)
	    # print(name, "% .2f" % score)

# picking and using the best performing model
# model = RandomForestRegressor()
model.fit(X_train, y_train)

# predicting with the best performing model
predict = model.predict(forecast)

# plotting the results into a graph
plt.figure(figsize=(15,8))
(df[:-90]['daily_avg']).plot(label='Historical Price')
(df[-91:]['daily_avg']).plot(label='Predicted Price')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Prediction on Daily Average Price of Bitcoin')
plt.legend()
plt.show()

