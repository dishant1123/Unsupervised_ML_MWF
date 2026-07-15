# forward propagation : 
"""
the  process of passing input data through the network to produce an output. informations moves in one  direction only.

input layer --> hidden layer --> output layer

output = activation_function(input * weight) + bias)

real life example  : 

studyhrs =8 
attendance = 90 % 
     |

multiply =input * weight
    add : z = input * weight + bias
    activation function :  a =f(z)
    
    predict  ----> 0 or 1
    |
output    ====>0.95   ====> 95% probability pass  
   
   
overall  : a =f(x *w + b)

"""

# ex :1 
import numpy  as np 

# single layer network :
"""
x=[2,3]
weights =[0.4,0.6]
b=0.5 

dot product : (2 * 0.4) + (3 *0.6) +0.5 = 3.1 
activation function  : sigmoid(3.1)   = 1 / 1 + e^-3.1  = 0.95  ====> pass 95 %  
"""

# ex :2 

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1 +np.exp(-x))

# input   data set  :4 samples 2 features 

x=np.array([
    [2,3],
    [4,5],
    [6,7],
    [8,9]
])
print("input data set : \n")
print(x)

# hidden layer  :
"""
input  neurons  : 2 neurons
hidden layer  : 3 neurons
"""

w1 =np.array([
    [0.2,0.4,0.1],
    [0.5,0.3,0.7]
])

b1 =np.array([[0.1,0.2,0.3]])

print("hidden layer weights : \n")
print(w1)

print("hidden layer bias : \n")
print(b1)

# forward propagation  :
"""
hidden layer :
"""
# step :1 z = x * w1 + b1

z1=np.dot(x,w1) + b1
print("z1 :\n",z1)
print("z1 shape :",z1.shape)

# step : 2  activation  function  : a = relu(z1)   
a1 = relu(z1)
print("a1 :\n",a1)  # shape (4,3) 

#output  layer  :
"""
hidden layer  : 3 neurons
output  neurons  : 1 neurons
"""
w2 =np.array([
    [0.4],
    [0.7],
    [0.6]
])

b2 =np.array([[0.5]])

print("output layer weights : \n")
print(w2)
print("output layer bias : \n")
print(b2)

# forward propagation  output layer pass:

# step : 3 z2 = a1 * w2 + b2

z2=np.dot(a1,w2) + b2
print("z2 :\n",z2)
print("z2 shape :",z2.shape)

a2 = sigmoid(z2)
print("a2 :\n",a2)  # 
print("a2 shape :",a2.shape)

# final prediction  :

print("final prediction :\n")
for i in range(len(a2)) :
    print(f"sample {i+1} prediction  : {a2[i][0] :.2f}")
    

# input(x) --->multiply by  w1 --->add bias b1 ---> linear output z1 ---> activation function (Relu) --->hidden layer output  a1  ---->multiply by w2  ---> add bias b2 --->linear output z2 ----> activation function (Sigmoid) ----> output layer output a2 ---> final prediction
