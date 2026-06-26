"""
Dimensionality Reduction is the process of reducing the number of features (columns) in a dataset while keeping as much important information as possible.

It converts many features into fewer new features.

=====================================================
Why do we need Dimensionality Reduction?

Imagine a dataset with 100 columns.

Problems:

Training becomes slow.
More memory is required.
Visualization becomes impossible.
Some columns contain duplicate information.
Noise affects model accuracy.

Instead of using all 100 features, PCA may reduce them to only 10 important features.

====================================================
simple  practical example : 

| Math | Science | Physics | Chemistry | Biology |
| ---- | ------- | ------- | --------- | ------- |
| 90   | 88      | 92      | 91        | 85      |

All science subjects are highly related.
Instead of using all 5 marks separately,

PCA creates
Science Performance = 89.8

Now instead of 5 columns, we have 1 new column.
Less data
Same information

=========================================
PCA :PCA (Principal Component Analysis) is a dimensionality reduction technique that converts correlated features into a smaller number of new uncorrelated features called Principal Components (PCs) while preserving as much information (variance) as possible.


Why Use PCA?
Reduce the number of features
Remove redundant information
Reduce noise
Improve model performance
Visualize high-dimensional data in 2D or 3D

Original Dataset
       ↓
Standardize Data
       ↓
Calculate Covariance Matrix
       ↓
Find Eigenvalues & Eigenvectors
       ↓
Sort Eigenvalues (Highest → Lowest)
       ↓
Select Top Principal Components
       ↓
Transform Data
       ↓
New Reduced Dataset
"""

# ex:1

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

"""data = pd.DataFrame({
    "height" :[150,160,170,180,190],
    "weight" :[50,60,70,80,90]
})

# covariance : 
scaler = StandardScaler()
scaler_fit = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaler_fit,columns=["height","weight"])

print(scaled_df)
print(scaled_df.cov())

pca=PCA(n_components=1)
pca_scaled_fit = pca.fit_transform(scaled_df)

scaled_df = pd.DataFrame(pca_scaled_fit,columns=["height"])
print(scaled_df)

print(pca.components_)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ * 100)
print(pca.explained_variance_ratio_.cumsum())

"""
# ex :2 iris dataset

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target

# print(df.sample)
# print(df.columns)
# print(df.head())
# print(df.isnull().sum())

# print(df.describe())

x= df.drop('species',axis=1)
# print(x.head())

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# print(x_scaled[:5])

# apply PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# print(x_pca)
# print(pca.components_)

# pca_dataframe : 

pca_df = pd.DataFrame(x_pca,columns=['PC1','PC2'])
print(pca_df)

# scatter plot : 

plt.figure(figsize=(8,6))

plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA on Iris Dataset')

plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3,random_state=42)

cluster = kmeans.fit_predict(x_pca)
pca_df['cluster'] = cluster
print(pca_df.head())

print(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
plt.scatter(
    pca_df['PC1'],
    pca_df['PC2'],
    c=pca_df['cluster'],
    s=70
)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='x',
    s=250,
    color='red',
    label='centroids'
)

plt.title('kmeans clustering after  PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.legend()
plt.grid(True)
plt.show()
