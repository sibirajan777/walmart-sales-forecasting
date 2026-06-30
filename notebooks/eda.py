from prophet import Prophet
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('../data/train.csv')
print("shape:",df.shape)
print("\nfirst 5 rows:\n",df.head())
print("\ncolumns info:\n",df.info())
print("\nmissing values:\n",df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])

sales_by_date=df.groupby('Date')['Weekly_Sales'].sum()  

sales_by_date.plot(figsize=(12, 6), title='Weekly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.plot(df['Date'],df['Weekly_Sales'], 'o', alpha=0.5, label='Individual Sales')
plt.legend()
plt.savefig('../outputs/sales_trend.png')
plt.show()

mean_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
print(mean_sales)

store_sales=df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
print(store_sales.head(5))

dept_sales=df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False)
print(dept_sales.head(5))

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month  
df['Week'] = df['Date'].dt.isocalendar().week
print(df[['Date','Year','Month','Week']].head())

df['IsHoliday'] = df['IsHoliday'].astype(int)
print(df['IsHoliday'].head()) 

train_data = df['Date'] < '2012-06-01'  
print("Train data shape:", df[train_data].shape)
test_data = df['Date'] >= '2012-06-01'
print("Test data shape:", df[test_data].shape)

prophet_df = sales_by_date.reset_index().rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
print(prophet_df.head())

model=Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12))

model.plot(forecast)
plt.title('Prophet Forecast of Weekly Sales')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.savefig('../outputs/prophet_forecast.png')
plt.show()