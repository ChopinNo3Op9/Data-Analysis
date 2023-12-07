# Importing Libraries and Modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import f_oneway

# Data Exploration and Preprocessing
df.info()
df.describe()
data.isnull().sum()
df_cleaned = df.drop_duplicates()
z_scores = stats.zscore(df['A'])
outliers = df[(z_scores > 3) | (z_scores < -3)]
Q1 = df['A'].quantile(0.25)
Q3 = df['A'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df['A'] >= lower_bound) & (df['A'] <= upper_bound)]

# Data Imputation and Transformation
scaler = MinMaxScaler(feature_range=(0, 1))
df['score_normalized'] = scaler.fit_transform(df[['score']])
df['score_standardized'] = zscore(df['score'])
max_abs_value = df['score'].abs().max()
num_digits = len(str(max_abs_value))
df['score_decimal_scaled'] = df['score'] / (10 ** num_digits)
df['score_log_transformed'] = np.log(df['score'])

df.dropna(inplace=True)
df.fillna(0, inplace=True)
imputer = SimpleImputer(strategy='mean') 
result_join = df1.join(df2, how='outer')
result_merge = pd.merge(df1, df2, on='key')
result_row = pd.concat([df1, df2], ignore_index=True)
filtered_df = df[df['Age'] > 30]
df_cleaned.apply(pd.to_numeric, errors='coerce')
data['ProductID'] = data['ProductID'].astype('category')
df.rename(columns={'A': 'New_A'}, inplace=True)
df.drop(columns=['New_Column1'], inplace=True) 
df['Last Update'] = pd.to_datetime(df['Last Update'])
df = pd.get_dummies(df, columns=['category'])
df.set_index('FY', inplace=True)
subset = df.loc[1:2, ['Name', 'Age']]
df['YoY ROCE Growth'] = df['Return on Common Equity'].pct_change() * 100
df['3-Year MA ROCE'] = df['Return on Common Equity'].rolling(window=3).mean()
total_sales_per_product = data.groupby('ProductID')['QuantitySold'].sum()
top_ytd_rtn = df.sort_values(by='YTD Rtn', ascending=False).head(5)
correlation_matrix = df.corr()

# Data Analysis and Visualization
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
data['DayOfWeek'] = data['SaleDate'].dt.day_name()

monday_sales = data[data['DayOfWeek'] == 'Monday']['SaleAmount']
sunday_sales = data[data['DayOfWeek'] == 'Sunday']['SaleAmount']

t_statistic, p_value = stats.ttest_ind(df['Age'], df['Salary']) 
t_statistic, p_value = stats.ttest_1samp(df['Age'], 20)
stat, p = f_oneway(monday_sales, sunday_sales)  # F-value, p-value
correlation_coef = np.corrcoef(df['Age'], df['Salary'])

# Machine Learning
# Linearity
predictions = model.predict(X)
plt.scatter(df['X'], df['Y'], label='Observed')
plt.scatter(df['X'], predictions, label='Predicted')
# Independence of Errors
residuals = model.resid
plt.plot(residuals)
# Homoscedasticity
plt.scatter(predictions, residuals)
# Normality of Residuals
sm.qqplot(residuals, line='s')
plt.title('Normality Assumption Check (Q-Q Plot)')
from scipy.stats import shapiro
shapiro_test_statistic, shapiro_p_value = shapiro(residuals)

X = df[['Age', 'Education']]  # Features (2D array for scikit-learn)
y = df['Salary']  # Target variable
model = LinearRegression()
model.fit(X, y)
print("Model Coefficients (Slope):", model.coef_)
print("Model Intercept:", model.intercept_)

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()

cv_scores = cross_val_score(model, X, y, cv=3)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))  # larger, better

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = svm.SVC()
model = LogisticRegression()
model = DecisionTreeClassifier()
model = KNeighborsClassifier(n_neighbors=5)
model = RandomForestClassifier(random_state=42)
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

r_squared = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
mse_cv = -scores.mean() 

# Data Visualization (Plots)
sns.pairplot(df[['Age', 'Education', 'Salary']])

data['SaleAmount'].hist()
sns.boxplot(x='ProductID', y='SaleAmount', data=data)
plt.scatter(x, y, color='r', marker='o', s=100, label='Scatter Plot')
plt.bar(categories, values, color='g', label='Bar Chart')
plt.hist(data, bins=3, color='purple', edgecolor='black', alpha=0.1, label='Histogram')
sns.countplot(x='ProductID', data=data)
plt.figure(figsize=(10, 6))
top_selling_products.plot(kind='bar')
average_sales_per_day.plot(kind='line')
plt.plot(df.index, df['YoY ROCE Growth'], marker='o', label='YoY ROCE Growth', color='b')
plt.axhline(0, color='r', linestyle='--', label='Zero Growth')
plt.title('Year-over-Year ROCE Growth')
plt.xlabel('Fiscal Year')
plt.ylabel('Growth Rate (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
