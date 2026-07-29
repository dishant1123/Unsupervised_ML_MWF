import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""
How do we choose them?

There is no fixed rule. We experiment.

For example:

Dataset Size	    Hidden Units (Typical)
Small (1k rows)	     8-32
Medium (10k rows)	16-64
Large (100k+ rows)	64-256
"""
# load data
df =pd.read_csv('deep learning/Churn_Modelling (1).csv')
# print(df.head())

"""
x ==> [ features]  ===> hidden layer  ===> 16 nueron  8 nueron  
y = output      ===> 1

64  ===> 32 ===> 16 ===> 8   ===>1  
"""

# remove  unnecessary columns :
df =df.drop(['RowNumber','CustomerId','Surname'],axis=1)
df=pd.get_dummies(df,drop_first=True)

# features : 
X=df.drop(["Exited"],axis=1)
y=df["Exited"]

# split :

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42)

# scale :
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build neural network :

model =tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=16, 
        activation='relu',
        input_shape=(X_train.shape[1],) #first layer must know how many input features for each training sample has
        ),
    tf.keras.layers.Dense(
            units=8, 
            activation='relu'
            
            ),
    tf.keras.layers.Dense(
        units=1, 
        activation='sigmoid'
        )
])

# summary :
print("model summary :",model.summary())

# compile :
model.compile(
    optimizer="adam",  # fast , accurate  adaptive momentum estimation  
    loss="binary_crossentropy",
    metrics=["accuracy"] 
)

# train model backpropagation happen automatically here:

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#evaluate model :

loss,accuracy = model.evaluate(X_test,y_test)
print("loss :",loss)
print("accuracy :",accuracy)

# prediction  : 
y_probability = model.predict(X_test)
y_prediction  = (y_probability > 0.5).astype(int)

# compare actual and predicted :

c =pd.DataFrame(
    {
        "actual" : y_test.values,
        "probability" : y_probability.flatten(),
        "prediction" : y_prediction.flatten()
    }
)

print(c.head(20))

# accuracy : 
acc =accuracy_score(y_test,y_prediction)
print("accuracy_score :",acc)

# confusion matrix :
conf_mat = confusion_matrix(y_test,y_prediction)
print("confusion matrix :",conf_mat)

# classification report :
print("classification report :",classification_report(y_test,y_prediction))

# test acc : 0.85 , proba : 0.91  , prediction  : customer will leave 

# predict new customer data:

new_customer_data = np.array([[
    650,
    45,
    5,
    80000,
    2,
    1,
    1,
    60000,
    0,
    0,
    1
]])

new_customer_data = scaler.transform(new_customer_data)
prediction  = model.predict(new_customer_data)

print("probability of leaving customer :",prediction[0][0])
if prediction > 0.5 :
    print("customer will leave")
else :
    print("customer will not leave")
    
    
"""
yrsexp  salary 

1       23000
2       34000

x_train ====> yrs of exp 
print(x_train.shape) 2,2   ====> 2 row  2 col    ===>2 samples 2 features 

area   bedroom   age  price 
1000   1         30   10000
1500   2         25   15000

shape ====> 2,3    ===>2 samples 3 features

input_shape [0]   ====> 2 sample 
input_shape [1]   ====> 3 features
"""
