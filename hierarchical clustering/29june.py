"""
hierarchical clustering : that groups similar data points together into clusters building a tree like hierarchy and its called as dendrogram. it not  specify to require the  number of clustering in advance. 

types of hierarchical clustering:

1.agglomerative clustering :   botton  to  up  
    ex : 
    
    name     marks  
    A         20 
    B         25
    C         90 
    D         96
    
    separate cluster  : [A] , [B] , [C] , [D]  ===>start each point 
                        |
                        merge closest  points  : [A,B]   [C]  [D]
                        |
                        merge closest  points  : [A,B]  [C,D]
                        |
                        merge closest  points  : [A,B,C]  [D]
                        |
                        merge closest  points  : [A,B,C,D]

                ABCD 
                  |
               [a]  [b]  [c]  [d]
                | 
               [a,b]  [c] [d]
                |
                [a,b]  [c,d]
                | 
                [a,b,c]  [d]
                |
                [a,b,c,d] 

2.divisive clustering : up  to  bottom
    ex : 
    
    name     marks  
    A         20 
    B         25
    C         90 
    D         96
    
    [A,B,C,D]
        |
    [A,B,C]  [D]
        |
    [A,B]  [C,D]
        |
    [A]  [B,C,D]
        |
    [A]  [B] [C]  [D]

linkage  : 
1.single  linkage :  the  distance  between  two  clusters  is  the  average  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

2. complete linkage :  the  distance  between  two  clusters  is  the  maximum  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

3. average linkage :  the  distance  between  two  clusters  is  the  average  of  the  distances  between  each  pair  of  points  in  the  two  clusters.

4. Ward's linkage :  the  distance  between  two  clusters  is  the  sum  of  the  squares  of  the  differences  of  each  pair  of  points  in  the  two  clusters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
data = {
    "student": ['A', 'B', 'C', 'D','E','F'],
    "Maths" : [20 ,22 ,80 ,82 ,25 ,78], 
    "Science": [25 ,24 ,81 ,84 ,28 ,79]
}

df = pd.DataFrame(data)
x =df[['Maths','Science']]
print("selected features\n",x)

# step :1  create  linkage  matrix

linkage_matrix = linkage(x, method='ward')
print("linkage matrix\n",linkage_matrix)

# draw dendrogram
plt.figure(figsize=(8,5))

dendrogram(linkage_matrix, labels=df['student'].values)
plt.title('Dendrogram')
plt.xlabel("studens")
plt.ylabel("distance")
plt.show()

# apply agglomerative clustering : 

model = AgglomerativeClustering(n_clusters=2, linkage='ward')

clusters = model.fit_predict(x)
df['cluster'] = clusters
print("cluster\n",df)

# plot cluster  : 

plt.scatter(
    df['Maths'],
    df['Science'],
    c=df['cluster'],
    s=150
)

for i in range(len(df)) : 
    plt.text(
        df['Maths'][i]+0.5,
        df['Science'][i]+0.5,
        df['student'][i],
        fontsize=12
    )

    
plt.title(' hirearchical clustering ')
plt.xlabel("Maths")
plt.ylabel("Science")
plt.grid(True)
plt.show()

