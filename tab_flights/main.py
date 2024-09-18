import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import results_kfold, EstimatorSelectionHelper
from plots import plot_heatmap
import lime
import shapely
import imblearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, \
    mean_absolute_error, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from timeit import default_timer as timer

pd.set_option('display.max_columns', None)

df = pd.read_csv(r"C:\Users\Administrator\Desktop\programming\datasets\business\939775908_T_ONTIME_REPORTING.csv")


df = df.drop(
    ['ARR_DEL15', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'TAXI_OUT', 'WHEELS_OFF', 'ORIGIN', 'WHEELS_ON', 'YEAR', 'TAXI_IN',
     'TAXI_OUT', 'CRS_ARR_TIME', 'ARR_TIME', 'DIVERTED', 'CRS_DEP_TIME', 'QUARTER', 'DEP_TIME', 'DIV_AIRPORT_LANDINGS',
     'DEST', 'OP_UNIQUE_CARRIER', 'CANCELLATION_CODE'], 1)


df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df = df.set_index('FL_DATE')
df = df.sort_values(by='FL_DATE')
df = df.dropna(how='all', axis='columns')
df = df.fillna(0)

corr = plot_heatmap(df)
corr.savefig('results/heatmap.jpg', bbox_inches='tight', dpi=150)

X = df.drop('CANCELLED', 1)
y = df['CANCELLED']

under = RandomUnderSampler(sampling_strategy=0.33)
X, y = under.fit_resample(X, y)
over = RandomOverSampler(sampling_strategy=1)
X, y = over.fit_resample(X, y)

models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier()
}

params = {
    'RandomForestClassifier': {'n_estimators': [16, 32]},
    'ExtraTreesClassifier': {'n_estimators': [16, 32]},
    'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
    'XGBClassifier': {'max_depth': [3, 5], 'min_child_weight': [3, 5]}
}

helper = EstimatorSelectionHelper(models, params)
helper.fit(X, y, scoring='f1', n_jobs=2)

helper.score_summary(sort_by='max_score')


regressors = [['DecisionTreeRegressor :', DecisionTreeRegressor()],
              ['RandomForestRegressor :', RandomForestRegressor()],
              ['ExtraTreesRegressor :', ExtraTreesRegressor()],
              ['GradientBoostingRegressor :', GradientBoostingRegressor()],
              ['XGBRegressor :', XGBRegressor()]]

model_results_reg = results_kfold(X, y, regressors, regression=True)
model_results_reg.to_csv('results/results_kfold_reg')

classifiers = [['RandomForestClassifier :', RandomForestClassifier()],
               ['ExtraTreesClassifier :', ExtraTreesClassifier()],
               ['GradientBoostingClassifier :', GradientBoostingClassifier()],
               ['XGBClassifier :', XGBClassifier()]]

model_results_clf = results_kfold(X, y, classifiers, regression=False)
model_results_reg.to_csv('results/results_kfold_clf')

model = XGBClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model.fit(X_train, y_train)

explainer = shapely.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shapely.summary_plot(shap_values, X_test)

shapely.summary_plot(shap_values, X_train, plot_type="bar")

