# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Display basic information about the datasets
print("\nCustomers Dataset Info:")
print(customers_df.info())
print("\nProducts Dataset Info:")
print(products_df.info())
print("\nTransactions Dataset Info:")
print(transactions_df.info())

# Basic statistics for numerical columns
print("\nCustomers Dataset Statistics:")
print(customers_df.describe())
print("\nProducts Dataset Statistics:")
print(products_df.describe())
print("\nTransactions Dataset Statistics:")
print(transactions_df.describe())

# Customer Analysis
plt.figure(figsize=(12, 6))
customers_df['Region'].value_counts().plot(kind='bar')
plt.title('Distribution of Customers by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Signup Date Analysis
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
customers_df['SignupMonth'] = customers_df['SignupDate'].dt.to_period('M')
monthly_signups = customers_df['SignupMonth'].value_counts().sort_index()

plt.figure(figsize=(15, 6))
monthly_signups.plot(kind='bar')
plt.title('Customer Signups by Month')
plt.xlabel('Month')
plt.ylabel('Number of Signups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Product Analysis
plt.figure(figsize=(12, 6))
products_df['Category'].value_counts().plot(kind='bar')
plt.title('Distribution of Products by Category')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price Distribution by Category
plt.figure(figsize=(12, 6))
products_df.boxplot(column='Price', by='Category', figsize=(12, 6))
plt.title('Price Distribution by Product Category')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Transaction Analysis
# Merge transactions with products to get product details
transactions_with_products = pd.merge(transactions_df, products_df, on='ProductID')

# Analyze sales by product category
category_sales = transactions_with_products.groupby('Category')['Quantity'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
category_sales.plot(kind='bar')
plt.title('Total Sales by Product Category')
plt.xlabel('Category')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly revenue trend
transactions_with_products['TransactionDate'] = pd.to_datetime(transactions_with_products['TransactionDate'])
monthly_revenue = transactions_with_products.groupby(transactions_with_products['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()

plt.figure(figsize=(15, 6))
monthly_revenue.plot(kind='line', marker='o')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print Key Business Insights
print("\nKey Business Insights:")
print("\n1. Regional Distribution:")
print(customers_df['Region'].value_counts().to_string())

print("\n2. Top Selling Categories:")
print(category_sales.head().to_string())

print("\n3. Average Transaction Value by Region:")
transactions_with_customers = pd.merge(transactions_with_products, customers_df, on='CustomerID')
avg_transaction_by_region = transactions_with_customers.groupby('Region')['TotalValue'].mean().sort_values(ascending=False)
print(avg_transaction_by_region.to_string())

print("\n4. Product Category Price Analysis:")
category_price_stats = products_df.groupby('Category')['Price'].agg(['mean', 'min', 'max']).round(2)
print(category_price_stats.to_string())

print("\n5. Customer Signup Trends:")
print(monthly_signups.head().to_string())
