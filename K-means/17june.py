"""
cluster : similar  data group divide . 

k-means : divide data into k groups .  k ===> number of  clusters.

data 
 | 
centroid  
  | 
update centroid 
 | 
assign data to centroid 
 | 
repeat
 |
model choose 
    | 
predict 

ex : data = [12,13,14,60,68,90]  k=2 
    group  1  : [12,13,14 ]
    group 2 : [60,68,90]
"""

import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

# ex :1 simple 
"""
data = {
    "age" :[12,14,18,45,50,55],
    "salary" : [1200,1800,3000,40000,45000,55000] 
    
}
df=pd.DataFrame(data)

kmeans = KMeans(n_clusters=2,random_state=42)  # k=2 

df["cluster"] = kmeans.fit_predict(df)

print(df)   
print("centroids",kmeans.cluster_centers_)
"""

# ex :2 using  dataset : 

df = pd.read_csv("K-means\Mall_Customers.csv")

X=df[["Annual Income (k$)","Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=5,random_state=42)  

df["cluster"] = kmeans.fit_predict(X)

print(df.head())
print("centroids",kmeans.cluster_centers_)  # centroids

# graphs  using  scatter plot

plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["cluster"]
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation using K-means")
plt.show()