"""
1.supervised learning : dataset , target  ex : house  price  prediction 
2.unsupervised learning : no label data , hidden pattern ex :customer segmentation 
3.reinforcement learning :rewards , punishments

ex : 

supervised learning :

studies hrs      result 
2                fail 
4                pass
6                pass

unsupervised learning :
color   
red  ====>hidden pattern  
red  ====> 
orange   ====> orange 
yellow   ====> banana 
  
unsupervised learning :

types : 

1. clustering : similar data  to make it cluster (group)
ex : 

customer_name    age    salary 
A                20     20000
B                22     22000
C                25     25000
D                40     90000
E                45     100000 

output :  2 cluster 
cluster : A,B,C    ====> low age  low  income  
cluster : D,E  ====> high age  high  income

     cluster : 1.K-means 2. DBSCAN 3. hierarchical clustering
     
2. association rule mining : find relationship between two variables

ex : customer purchase daily products : 
transaction : 

bread ,milk 
bread,milk,butter
bread,milk,cheese 


3. demension reduction : reduce dimension   important features 
    PCA : principal component analysis
ex : 
    height  weight   BMI  age   salary 

difference  between  supervised vs  unsupervised learning
                 supervised               unsupervised
1. data label       yes                     no
2. target variable  yes                     no
3.                 input+output             input 
4. ex :             house price prediction   customer segmentation

when to use  unsupervised learning  ? 

1. no label data , no target variable
2. grouping  for similar data , identify the  hidden  pattern 

Anomaly Detection : 

hamza cr credit card transactions : 

500    ===> 12/6   ===>petrol 
600    ===> 13/6   ===>food 
40000  ====> 14/6  fraud 


"""
