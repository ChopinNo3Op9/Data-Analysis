import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data.head()
data.info()
data.describe()

df.isnull().sum()
df = df.drop_duplicates()
z_score = stats.zscore(df['A'])
outliers = df[(z_score > 3) | (z_score < 3)]
Q1 = df['A'].quantile(0.25)
Q3
IQR = Q3 - Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
df = df[(df['A'] >= lower_bound) & (df['A'] <= upper_bound)]

scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df[])
df = zscore(df[''])
max_value = df[].abs.max()
ndigits = len(str(df[].abs().max()))
df[] = df[]/(10**ndigits)
df = np.log(df[])

df.dropna(inplace=True)
df.fillna(inplace=True)
imputer = SimpleImputer(strategy='mean')
df.join(df2, how='outer')
pd.merge(df1, df2, 'key')
pd.concat([df1, df2])
df[df[] > 30]
df[] = df[].astype('category')
df.rename({"a":"A"}, inplace=True)
df.drop(['A'], inplace=True)
df['time'] = pd.to_datetime(df['time'])
df[] = pd.get_dummies(df, columns=['category'])
df.set_index('date', inplace=True)
subset = df.loc[1:2, ['A', 'B']]
df[] = df[].pct_change()*100
df[] = df[].rolling(windows=3).mean()
s = df.groupby("").sum()
d = df.sort_values(by='T', ascending=True).head()
df.corr()

sns.heatmap(cor_matrix, annot=True, linewwidths=5)
dt.day_name()

monday_sales = df[df['Dateofweek'] == 'Monday']['SaleAmount']
sunday_sales = df[df[] == 'Sunday']['SaleAmount']

t, p = stats.ttest_ind(x, y)
f, p = f_oneway(x, y)
np.corrcoef(x, y)



