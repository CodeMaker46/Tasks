import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import gdown
from datetime import datetime

def download_data():
    urls = {
        'customers': 'https://drive.google.com/uc?id=1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE',
        'products': 'https://drive.google.com/uc?id=1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0',
        'transactions': 'https://drive.google.com/uc?id=1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF'
    }
    
    for name, url in urls.items():
        output = f'{name}.csv'
        gdown.download(url, output, quiet=False)

class LookalikeModel:
    def __init__(self):
        self.customers_df = None
        self.products_df = None
        self.transactions_df = None
        self.customer_features = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        self.customers_df = pd.read_csv('customers.csv')
        self.products_df = pd.read_csv('products.csv')
        self.transactions_df = pd.read_csv('transactions.csv')
        
        self.customers_df['SignupDate'] = pd.to_datetime(self.customers_df['SignupDate'])
        self.transactions_df['TransactionDate'] = pd.to_datetime(self.transactions_df['TransactionDate'])
        
    def prepare_features(self):
        current_date = pd.Timestamp('2025-01-27')
        self.customers_df['DaysOnPlatform'] = (current_date - self.customers_df['SignupDate']).dt.days
        
        customer_transactions = self.transactions_df.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'Quantity': 'sum',
            'TotalValue': 'sum',
            'TransactionDate': ['min', 'max']
        }).reset_index()
        
        customer_transactions.columns = ['CustomerID', 'TransactionCount', 'TotalQuantity', 
                                      'TotalSpent', 'FirstPurchaseDate', 'LastPurchaseDate']
        
        customer_transactions['AvgTransactionValue'] = (
            customer_transactions['TotalSpent'] / customer_transactions['TransactionCount']
        )
        
        customer_transactions['PurchaseFrequencyDays'] = np.where(
            customer_transactions['TransactionCount'] > 1,
            (customer_transactions['LastPurchaseDate'] - customer_transactions['FirstPurchaseDate']).dt.days / 
            (customer_transactions['TransactionCount'] - 1),
            0
        )
        
        category_preferences = self.calculate_category_preferences()
        
        self.customer_features = pd.merge(
            self.customers_df,
            customer_transactions,
            on='CustomerID',
            how='left'
        )
        
        self.customer_features = pd.merge(
            self.customer_features,
            category_preferences,
            on='CustomerID',
            how='left'
        )
        
        numeric_columns = ['TransactionCount', 'TotalQuantity', 'TotalSpent', 
                         'AvgTransactionValue', 'PurchaseFrequencyDays']
        self.customer_features[numeric_columns] = self.customer_features[numeric_columns].fillna(0)
        
        category_columns = [col for col in self.customer_features.columns if col.startswith('Category_')]
        self.customer_features[category_columns] = self.customer_features[category_columns].fillna(0)
        
        date_columns = ['FirstPurchaseDate', 'LastPurchaseDate']
        for col in date_columns:
            self.customer_features[col] = pd.to_datetime(
                self.customer_features[col].fillna(self.customer_features['SignupDate'])
            )
        
        self.feature_columns = [
            'DaysOnPlatform',
            'TransactionCount',
            'TotalQuantity',
            'TotalSpent',
            'AvgTransactionValue',
            'PurchaseFrequencyDays'
        ]
        
        self.feature_columns.extend(category_columns)
        
        self.customer_features_scaled = self.scaler.fit_transform(
            self.customer_features[self.feature_columns].fillna(0).values
        )
        
    def calculate_category_preferences(self):
        trans_with_categories = pd.merge(
            self.transactions_df,
            self.products_df[['ProductID', 'Category']],
            on='ProductID'
        )
        
        category_spending = trans_with_categories.pivot_table(
            index='CustomerID',
            columns='Category',
            values='TotalValue',
            aggfunc='sum',
            fill_value=0
        )
        
        category_spending.columns = [f'Category_{col}' for col in category_spending.columns]
        
        return category_spending.reset_index()
        
    def get_customer_profile(self, customer_id):
        customer = self.customer_features[
            self.customer_features['CustomerID'] == customer_id
        ].iloc[0]
        
        category_cols = [col for col in self.customer_features.columns if col.startswith('Category_')]
        categories = [(col.replace('Category_', ''), customer[col]) 
                     for col in category_cols if customer[col] > 0]
        categories.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'CustomerID': customer['CustomerID'],
            'CustomerName': customer['CustomerName'],
            'Region': customer['Region'],
            'SignupDate': customer['SignupDate'].strftime('%Y-%m-%d'),
            'DaysOnPlatform': int(customer['DaysOnPlatform']),
            'TransactionCount': int(customer['TransactionCount']),
            'TotalQuantity': int(customer['TotalQuantity']),
            'TotalSpent': float(customer['TotalSpent']),
            'AvgTransactionValue': float(customer['AvgTransactionValue']),
            'PurchaseFrequencyDays': float(customer['PurchaseFrequencyDays']),
            'TopCategories': categories[:3]
        }
        
    def find_similar_customers(self, customer_id, n_recommendations=3):
        if not isinstance(customer_id, str) or not customer_id.startswith('C'):
            customer_id = f'C{int(customer_id):04d}'
            
        if customer_id not in self.customer_features['CustomerID'].values:
            raise ValueError(f"Customer ID {customer_id} not found in the database")
        
        target_profile = self.get_customer_profile(customer_id)
            
        customer_idx = self.customer_features[
            self.customer_features['CustomerID'] == customer_id
        ].index[0]
        
        similarities = cosine_similarity(
            self.customer_features_scaled[customer_idx].reshape(1, -1),
            self.customer_features_scaled
        )[0]
        
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations + 1]
        
        recommendations = []
        for idx in similar_indices:
            customer_id = self.customer_features.iloc[idx]['CustomerID']
            customer_profile = self.get_customer_profile(customer_id)
            customer_profile['Similarity'] = float(similarities[idx])
            recommendations.append(customer_profile)
        
        return target_profile, recommendations

def main():
    download_data()
    model = LookalikeModel()
    model.load_data()
    model.prepare_features()
    target_profile, similar_customers = model.find_similar_customers(5)

if __name__ == "__main__":
    main()
