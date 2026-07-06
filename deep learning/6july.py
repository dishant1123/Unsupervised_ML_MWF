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

=0.5 * 5 + 0.8 * 8 +0.1= 7.2 


"""
