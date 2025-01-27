import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('default')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
customers_df['SignupYear'] = customers_df['SignupDate'].dt.year
yearly_signups = customers_df['SignupYear'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
yearly_signups.plot(kind='bar')
plt.title('Customer Signups by Year')
plt.xlabel('Year')
plt.ylabel('Number of Signups')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
customers_df['Region'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Customer Distribution by Region')
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
products_df.boxplot(column='Price', by='Category')
plt.title('Price Distribution by Product Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

category_stats = products_df.groupby('Category')['Price'].agg(['mean', 'min', 'max', 'count']).round(2)

transactions_analysis = transactions_df.merge(products_df, on='ProductID').merge(customers_df, on='CustomerID')

transactions_analysis['TransactionDate'] = pd.to_datetime(transactions_analysis['TransactionDate'])
monthly_sales = transactions_analysis.groupby(transactions_analysis['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()

plt.figure(figsize=(15, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Value')
plt.grid(True)
plt.tight_layout()
plt.show()

category_sales = transactions_analysis.groupby('Category').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).sort_values('TotalValue', ascending=False)

regional_performance = transactions_analysis.groupby('Region').agg({
    'TransactionID': 'count',
    'TotalValue': ['sum', 'mean']
}).round(2)
regional_performance.columns = ['Transaction_Count', 'Total_Sales', 'Average_Transaction_Value']

avg_transaction_value = transactions_df['TotalValue'].mean()
total_revenue = transactions_df['TotalValue'].sum()
total_customers = len(customers_df)
total_products = len(products_df)
transactions_per_customer = len(transactions_df) / len(customers_df)

category_analysis = products_df.groupby('Category').agg({
    'ProductID': 'count',
    'Price': 'mean'
}).round(2)

recent_customers = customers_df[customers_df['SignupDate'] >= '2024-01-01'].shape[0]

top_products = transactions_analysis.groupby('ProductName').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).sort_values('TotalValue', ascending=False).head()
