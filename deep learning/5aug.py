"""
functional API syntax : 
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Concatenate

"""inputs = Input(shape=(30,))

X =Dense(16, activation='relu')(inputs)
X=Dense(8, activation='relu')(X)
outputs = Dense(1, activation='sigmoid')(X) 

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
"""

# binary classification  using functional API :

'''data =load_breast_cancer()

# features 
X=data.data 
y=data.target

# split data :
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42)

# scaler : 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build model : using funcational API :

inputs = Input(shape=(30,))

x=Dense(16, activation='relu')(inputs)
x=Dense(8, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model =Model(inputs=inputs, outputs=outputs) 

"""model =tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=16, 
        activation='relu',
        input_shape=(X_train.shape[1],) #first layer must know how many input features for each training sample has
        ),
])
"""
# model summary :
print("summary :",model.summary())

# compile model :
model.compile(
    optimizer="adam",  # fast , accurate  adaptive momentum estimation
    loss = "binary_crossentropy",
    metrics=["accuracy"]
)

# model train/fit :
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
)

# evaluate model :

loss ,accuracy = model.evaluate(X_test,y_test)
print("loss :",loss)
print("accuracy :",accuracy)
'''

# multi input network :
"""
bank wants to predict whether a customer will take a loan or not.

inputs:1  personal details 
====> age 
====> salary 
====> Experience

inputs:2 Credit details
====> credit score
====> previous loans
====>EMI

functional API  : 

personal details / data 
     | 
    dense(16)
     | 
      ---------|
            concatenate
     ----------|
     |   
credit data 
    | 
    dense(16)
     | 
               |
            dense(8)
                |
            output 

"""

"""personal_input = Input(shape=(3,),name="personal_input")
credit_input = Input(shape=(3,),name="credit_input")

personal_branch = Dense(8,activation='relu')(personal_input)
certain_branch = Dense(8,activation='relu')(credit_input)

merged_branch = Concatenate()([personal_branch,certain_branch])

x=Dense(8,activation='relu')(merged_branch)
outputs = Dense(1, activation='sigmoid')(x)

model =Model(inputs=[personal_input,credit_input], outputs=outputs)

model.summary()
"""

# ex :2 HR  predict : employee will leave or not , expected salary 

"""
        employess_data 
        | 
        dense(16)
        |
        dense(8)
        |
    -----------
    |         |
leave        salary

x=dense(32,activation='relu')(inputs)
x=dense(16,activation='relu')(x)

attrition = Dense(1, activation='sigmoid')(x)
salary = Dense(1, activation='sigmoid')(x)

model =Model(inputs=inputs, outputs=[attrition,salary])

       
"""
    