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