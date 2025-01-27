import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import requests
import io
from datetime import datetime

def download_file_from_google_drive(url):
    file_id = url.split('/d/')[-1].split('/view')[0]
    
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    response = requests.get(download_url)
    return pd.read_csv(io.StringIO(response.content.decode('utf-8')))

print("Downloading datasets...")
customers_url = "https://drive.google.com/file/d/1bu_--mo79VdUG9oin4ybfFGRUSXAe-WE/view?usp=sharing"
products_url = "https://drive.google.com/file/d/1IKuDizVapw-hyktwfpoAoaGtHtTNHfd0/view?usp=sharing"
transactions_url = "https://drive.google.com/file/d/1saEqdbBB-vuk2hxoAf4TzDEsykdKlzbF/view?usp=sharing"

try:
    customers_df = download_file_from_google_drive(customers_url)
    products_df = download_file_from_google_drive(products_url)
    transactions_df = download_file_from_google_drive(transactions_url)
    print("Successfully downloaded all datasets!")
except Exception as e:
    print(f"Error downloading datasets: {str(e)}")
    exit(1)

print("\nCustomers DataFrame columns:", customers_df.columns.tolist())
print("\nTransactions DataFrame columns:", transactions_df.columns.tolist())

print("\nPreprocessing customer data...")
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
customers_df['CustomerAge'] = (pd.Timestamp.now() - customers_df['SignupDate']).dt.days

le = LabelEncoder()
customers_df['RegionEncoded'] = le.fit_transform(customers_df['Region'])

print("\nProcessing transaction data...")
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
latest_date = transactions_df['TransactionDate'].max()

rfm = transactions_df.groupby('CustomerID').agg({
    'TransactionDate': lambda x: (latest_date - x.max()).days,
    'TransactionID': 'count',
    'TotalValue': 'sum'
}).rename(columns={
    'TransactionDate': 'Recency',
    'TransactionID': 'Frequency',
    'TotalValue': 'Monetary'
})

transaction_features = transactions_df.groupby('CustomerID').agg({
    'ProductID': 'nunique',
    'Quantity': ['sum', 'mean'],
    'Price': ['mean', 'std']
}).reset_index()

transaction_features.columns = ['CustomerID', 'unique_products', 
                              'total_quantity', 'avg_quantity',
                              'avg_price', 'price_std']

df = customers_df.merge(rfm.reset_index(), on='CustomerID', how='left')
df = df.merge(transaction_features, on='CustomerID', how='left')

df = df.fillna(0)

features = ['CustomerAge', 'RegionEncoded', 'Recency', 'Frequency', 'Monetary',
           'unique_products', 'total_quantity', 'avg_quantity', 'avg_price', 'price_std']

print("\nSelected features for clustering:", features)

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print("\nFinding optimal number of clusters...")
db_scores = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    print(f"Testing k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    db_score = davies_bouldin_score(X_scaled, clusters)
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    db_scores.append(db_score)
    silhouette_scores.append(silhouette_avg)
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")

optimal_k = k_range[np.argmin(db_scores)]
print(f"\nOptimal number of clusters based on Davies-Bouldin Index: {optimal_k}")

print("Performing final clustering...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

final_db_score = davies_bouldin_score(X_scaled, df['Cluster'])
final_silhouette_score = silhouette_score(X_scaled, df['Cluster'])
print(f"Final Davies-Bouldin Index: {final_db_score:.4f}")
print(f"Final Silhouette Score: {final_silhouette_score:.4f}")

print("\nCreating visualizations...")
plt.figure(figsize=(10, 6))
plt.plot(k_range, db_scores, 'bo-', label='Davies-Bouldin Index')
plt.plot(k_range, silhouette_scores, 'ro-', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('Clustering Metrics vs Number of Clusters')
plt.legend()
plt.grid(True)
plt.savefig('clustering_metrics.png')
plt.close()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Customer Segments Visualization (PCA)')
plt.colorbar(scatter, label='Cluster')
plt.savefig('cluster_visualization_pca.png')
plt.close()

feature_importance = np.abs(pca.components_)
feature_importance = feature_importance / feature_importance.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 6))
sns.heatmap(feature_importance, annot=True, fmt='.2f', 
            xticklabels=features, yticklabels=['PC1', 'PC2'])
plt.title('Feature Importance in Principal Components')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

cluster_profiles = df.groupby('Cluster')[features].mean()
print("\nCluster Profiles:")
print(cluster_profiles)

cluster_profiles.to_csv('cluster_profiles.csv')

cluster_profiles_normalized = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

fig = plt.figure(figsize=(20, 15))
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)

for i in range(optimal_k):
    values = cluster_profiles_normalized.iloc[i].values
    values = np.concatenate((values, [values[0]]))
    angles_plot = np.concatenate((angles, [angles[0]]))
    
    ax = fig.add_subplot(2, (optimal_k+1)//2, i+1, projection='polar')
    ax.plot(angles_plot, values)
    ax.fill(angles_plot, values, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(features, size=8)
    ax.set_title(f'Cluster {i} Profile')

plt.tight_layout()
plt.savefig('cluster_profiles_radar.png')
plt.close()

print("\nAnalysis complete! Check the generated CSV files and visualizations for results.")
