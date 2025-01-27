# Customer Segmentation Analysis Report

## 1. Methodology

### 1.1 Data Preprocessing
- **Customer Profile Features**:
  - Customer Age (calculated from SignupDate)
  - Region (encoded)
  
- **Transaction-based Features**:
  - RFM Analysis:
    - Recency: Days since last purchase
    - Frequency: Number of transactions
    - Monetary: Total value of purchases
  - Product Diversity:
    - Number of unique products
    - Total quantity purchased
    - Average quantity per transaction
  - Price Behavior:
    - Average price per item
    - Price standard deviation (purchase variability)

### 1.2 Clustering Algorithm
- Algorithm: K-means Clustering
- Feature Scaling: StandardScaler
- Number of clusters tested: 2 to 10

## 2. Clustering Results

### 2.1 Optimal Clustering Configuration
- **Number of clusters**: 7 (determined by minimum Davies-Bouldin Index)
- **Features used**: 10 features combining both profile and transaction information

### 2.2 Clustering Metrics
1. **Davies-Bouldin Index**: 1.5415
   - Lower values indicate better clustering
   - Best score among all tested cluster numbers (2-10)

2. **Silhouette Score**: 0.1493
   - Range: [-1, 1], higher values indicate better clustering
   - Indicates moderate cluster separation

### 2.3 Cluster Characteristics

#### Cluster 0 (High-Value Active Customers)
- Size: 10 customers
- Characteristics:
  - High frequency (9.60 transactions)
  - Highest monetary value ($6,580.38)
  - Very recent purchases (17.70 days)
  - Most diverse product selection (9.10 products)

#### Cluster 1 (New Regular Customers)
- Size: 37 customers
- Characteristics:
  - Newest customers (301.97 days)
  - Moderate frequency (5.57 transactions)
  - Good monetary value ($3,127.30)
  - Recent purchases (51.19 days)

#### Cluster 2 (Inactive Customers)
- Size: 12 customers
- Characteristics:
  - Very low frequency (0.92 transactions)
  - Lowest monetary value ($445.45)
  - Least recent purchases (222.92 days)
  - Minimal product diversity (0.92 products)

#### Cluster 3 (Mid-Value Occasional Customers)
- Size: 26 customers
- Characteristics:
  - Lower frequency (3.38 transactions)
  - Good monetary value ($3,284.14)
  - Moderate recency (78.08 days)
  - Limited product diversity (3.27 products)

#### Cluster 4 (Long-term Low-Value Customers)
- Size: 53 customers
- Characteristics:
  - Long-term customers (730.74 days)
  - Low frequency (3.57 transactions)
  - Lower monetary value ($1,935.87)
  - Limited product diversity (3.53 products)

#### Cluster 5 (High-Value Regular Customers)
- Size: 36 customers
- Characteristics:
  - High frequency (6.64 transactions)
  - High monetary value ($5,107.17)
  - Good recency (69.11 days)
  - Good product diversity (6.42 products)

#### Cluster 6 (Loyal High-Value Customers)
- Size: 26 customers
- Characteristics:
  - Longest-term customers (839.62 days)
  - High frequency (6.58 transactions)
  - High monetary value ($5,049.60)
  - Good product diversity (6.38 products)

## 3. Visualization Suite

### 3.1 Cluster Quality Visualization
- `clustering_metrics.png`: Shows both Davies-Bouldin Index and Silhouette Score across different cluster numbers
- Helps validate the optimal number of clusters (k=7)

### 3.2 Cluster Distribution Visualization
- `cluster_visualization_pca.png`: 2D visualization using PCA
- Shows cluster separation and distribution
- Colors indicate different clusters

### 3.3 Feature Analysis
- `feature_importance.png`: Heatmap showing feature importance in principal components
- Helps understand which features contribute most to cluster formation

### 3.4 Cluster Profiling
- `cluster_profiles_radar.png`: Radar charts showing characteristics of each cluster
- Helps understand the unique traits of each customer segment

### 3.5 Cluster Size Distribution
- `cluster_sizes.png`: Bar chart showing the number of customers in each cluster
- Shows relatively balanced cluster sizes (10-53 customers per cluster)

## 4. Business Implications

### 4.1 Marketing Strategies
1. **Cluster 0 (High-Value Active)**: 
   - VIP treatment
   - Early access to new products
   - Exclusive offers

2. **Cluster 1 (New Regular)**:
   - Engagement programs
   - Product recommendations
   - Loyalty program enrollment

3. **Cluster 2 (Inactive)**:
   - Re-engagement campaigns
   - Special comeback offers
   - Feedback surveys

4. **Clusters 3 & 4 (Occasional/Low-Value)**:
   - Up-selling opportunities
   - Product education
   - Value-based promotions

5. **Clusters 5 & 6 (High-Value Regular/Loyal)**:
   - Retention programs
   - Premium services
   - Referral programs

### 4.2 Recommendations
1. **Immediate Actions**:
   - Re-engagement campaign for Cluster 2
   - VIP program for Clusters 0, 5, and 6
   - Activation campaign for Clusters 3 and 4

2. **Long-term Strategies**:
   - Develop loyalty program tiers
   - Implement personalized marketing
   - Regular monitoring of cluster transitions

## 5. Technical Implementation

The implementation is available in two files:
1. `customer_segmentation.py`: Main clustering implementation
2. `requirements.txt`: Required Python packages

### 5.1 Key Dependencies
- pandas: Data manipulation
- scikit-learn: Clustering and metrics
- matplotlib/seaborn: Visualization
- numpy: Numerical operations

### 5.2 Output Files
1. `clustering_metrics.png`: Clustering quality metrics
2. `cluster_visualization_pca.png`: PCA-based visualization
3. `feature_importance.png`: Feature importance heatmap
4. `cluster_profiles_radar.png`: Cluster characteristics
5. `cluster_sizes.png`: Cluster size distribution
6. `customer_segments.csv`: Customer cluster assignments
7. `cluster_profiles.csv`: Detailed cluster statistics

## 6. Reproducibility

The analysis can be reproduced by:
1. Installing required packages: `pip install -r requirements.txt`
2. Running the script: `python customer_segmentation.py`

## 7. Future Improvements

1. **Algorithm Exploration**:
   - Try other clustering algorithms (DBSCAN, Hierarchical)
   - Experiment with different feature combinations

2. **Feature Engineering**:
   - Include seasonal purchase patterns
   - Add product category preferences
   - Consider customer lifetime value

3. **Validation**:
   - Cross-validation of clusters
   - A/B testing of marketing strategies per cluster
