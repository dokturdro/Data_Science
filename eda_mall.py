import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv(r"C:\Users\Administrator\Desktop\datasets\business\Mall_Customers.csv")
print(df.columns.values)
print(df.head())

sns.countplot(df["Gender"].dropna())
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df[features].hist(figsize=(10, 6))
df_corr = df.corr()
sns.heatmap(df_corr, cbar=False, annot=True)

sns.jointplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, kind='scatter')
sns.boxplot(data=df.ix[:,1:3])

plt.show()