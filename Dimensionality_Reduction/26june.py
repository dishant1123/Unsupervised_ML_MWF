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

data = pd.DataFrame({
    "height" :[150,160,170,180,190],
    "weight" :[50,60,70,80,90]
})

# covariance : 
scaler = StandardScaler()
scaler_fit = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaler_fit,columns=["height","weight"])

print(scaled_df)
print(scaled_df.cov())


