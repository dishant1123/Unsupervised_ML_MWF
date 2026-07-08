"""
perceptron : It takes multiple inputs, assigns weights to each input, calculates a weighted sum, applies an activation function, and produces an output.
Think of it as a mathematical decision maker.

Real-Life Example :
Suppose a bank wants to decide whether to approve a loan.

Inputs:
Salary
Credit Score
Existing Loan

Output:
Approve Loan (1)
Reject Loan (0)

The perceptron learns which factors are more important.

          x1 -------\
                     \
          x2 ---------> ( Weighted Sum ) --> Activation --> Output
                     /
          x3 -------/

        z=x1w1+  x2w2 + x3w3  +b

hrs study     hrs sleep     pass 
2               5            0      w1 =0 ,w2=0 , b =0 
3               6            0
5               8            1
6               7            1 

z=x1w1+  x2w2  +b    # x1 =2   x2 =5 
  2*0  + 5*0  + 0  = 0    actual output  = 0  predict =0 


z=x1w1+  x2w2  +b    # x1 =5   x2 =8 
  5*0  + 8*0  + 0  = 0    actual output  = 1  predict =0    answer =  -1  

updated : 

wi = n * (y - y^) * xi 

w1 =0.1 * (1- 0)*5  =0.5 
w2 = 0.1 * (1- 0)*8  =0.8
b=  b + n * (y - y^)  = 0 + 0.1 * (1-0)= 0.1 

x1 =5   x2 = 8 

=0.5 * 0.1 + 0.1 * 8 +0= 7.2  ==> 0.05 + 0.8 +0


"""
# solving  this  using numpy  : 

"""
import numpy as np

x=np.array([
    [2,5],
    [5,8],
    [3,6],
    [6,7]
])

y=np.array([0,0,1,1])
w=np.array([0.0,0.0]) 
b=0.0
lr = 0.1 

print("intial weights : ",w)
print("initial bias : ",b)

for i in range(len(x)):
    X = x[i]
    z=np.dot(X,w) + b
    
    y_pred = 1  if z >=0 else 0 
    
    print("input : ",X)
    print("actual output : ",y[i])
    print("weightd sum : ",z)
    print("predicted output : ",y_pred)
    
    error = y_pred -y[i]
    print("error : ",error)
    
    # weight update  :  
    w =w + lr * error *X 
    b=b +lr * error
    
    print("updated weights : ",w)
    print("updated bias : ",b)
"""

# multi layer perceptron : 

import numpy as np

"""
study      attendance     assignment     pass 
2            60             4            0 
3            65             5            0
5            80             8            1
6            90             9            1

input  :   st.hrs , atted, assig_score 

hidden layer  :
    1. hard -working  student 
    2. regular student
    3. good assignment performance 
    
output : pass / fail 

"""
"""x=np.array([5,80,8])    # 5 hrs , 80  attend  8 assign 

# input  ---> hidden weight (3 inputs * 2 hidden  neurons)
w1 =np.array([
    [0.2,0.5],
    [0.3,0.4],
    [0.6,0.1]
])

# hidden bias : 
b1 =np.array([0.5,0.2])

# hidden ---> output  weight (2 hidden * 1 output)
w2 =np.array([
    [0.7],
    [0.8]
])

# output  bias : 

b2 =np.array([0.1])

hidden_input  = np.dot(x,w1) + b1 
hidden_output = np.maximum(0,hidden_input)

output = np.dot(hidden_output,w2) + b2

print("hidden layer  input  : ",hidden_input)
print("hidden layer  output : ",hidden_output)
print("output : ",output)
"""
"""
[5,80 ,8]   ===> multiply with w1 

hidden cal : 

neuron :1 

= 5 * 0.2 + 80 * 0.3  + 8 *0.6 +0.5 
  x1 *w1  + x2 * w2   + x3 * w3  + b 
  
neuron :2 

=5 * 0.5 + 80 * 0.4  + 8 *0.1 +0.2 

| 

np.max(0,x)   

| 
output  

hidden output *w2 +b 

| 
final output

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


students = pd.DataFrame({
    "Hours":[1,2,3,4,5,6,7,8],
    "Marks":[35,40,50,60,70,78,85,92]
})


x = students["Hours"].values

# Normalize
x = (x - np.mean(x)) / np.std(x)

sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.figure(figsize=(8,5))
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")
plt.plot(x, relu, label="ReLU")
plt.title("Activation Functions on Student Hours Dataset")
plt.xlabel("Normalized Study Hours")
plt.ylabel("Activation Output")
plt.legend()
plt.grid(True)
plt.show()