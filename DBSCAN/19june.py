"""
DBSCAN : density based clustering algorithum

eps        = How close points must be.
min_samples = How many close friends are needed to form a group.

"""
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ex:1 
"""data = {
    "x ":[1,1,2,3,5,7,8,20],
    "y" :[1,2,3,4,4,8,8,20] 
    
}

X= pd.DataFrame(data)
db = DBSCAN(eps=4,min_samples=2)
X['cluster'] = db.fit_predict(X)
print(X)
"""
# ex:2 
# read 
df = pd.read_csv("K-means\Mall_Customers.csv")

# col select : 
x=df[['Annual Income (k$)','Spending Score (1-100)']]
# print(x.head())

# scale :
scale = StandardScaler() 
x_scaled = scale.fit_transform(x)


# model : 
db = DBSCAN(eps=0.4,min_samples=3)
x['cluster'] =db.fit_predict(x_scaled)

print(x.tail(20))

# plot  : 

plt.scatter(
    x['Annual Income (k$)'],
    x['Spending Score (1-100)'],
    c=x['cluster'],
    marker="o"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("mall customer segmentation using DBSCAN")
plt.show()

"""
neighbors = NearestNeighbors(n_neighbors=5)
n =neighbors.fit(x_scaled)

distance ,indices = n.kneighbors(x_scaled)
distance = np.sort(distance[:,2])

# plot : 

plt.plot(distance)
plt.xlabel("index")
plt.ylabel("distance")
plt.title("distance")
plt.show()
"""
"""
ex : 
1.2  1.4     1.5 
1.23  1.43   1.56
1.21  1.00   1.2

"""
