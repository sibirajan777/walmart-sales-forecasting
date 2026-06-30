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
