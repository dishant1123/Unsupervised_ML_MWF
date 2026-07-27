"""

1.read_csv 
2. missing  
3. fillna 
4. x  ===> feature,y ===> target 
5. scale 
6. split
7. model build  : neural network
8. model compile
9. summary  
10. model.fit 
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

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

model =Sequential([
    Dense(
        128,
        kernel_regularizer=l2(0.001),
        input_shape=(X_train.shape[1],) 
        ),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    
    Dropout(0.30),
    Dense(
            64, 
            activation='relu',
            kernel_regularizer=l2(0.001)
            ),
    Dropout(0.30),
    
    Dense(
                32, 
                activation='relu',
                kernel_regularizer=l2(0.001)
                ),
    Dense(
        units=1, 
        activation='sigmoid'
        )
])

# compile :
model.compile(
    optimizer="adam",  # fast , accurate  adaptive momentum estimation  
    loss="binary_crossentropy",
    metrics=["accuracy"] 
)
# summary :
print("model summary :",model.summary())

# early stopping :
early_stopping =EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# train model backpropagation happen automatically here:
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

#evaluate model :

loss,accuracy = model.evaluate(X_test,y_test)
print("loss :",loss)
print("accuracy :",accuracy)

# prediction  : 
y_probability = model.predict(X_test)
y_prediction  = (y_probability > 0.5).astype(int)

print("first 10 rows of prediction :",y_prediction[:10])

# compare actual and predicted :
c =pd.DataFrame(
    {
        "actual" : y_test.values,
        "prediction" : y_prediction.flatten()
    }
)

print(c.head(20))

# loss : 0.35  acc : 0.85 

"""
conclusion :

0.85 ====> model correctly predicted the churn status  for  appox 86 out of every 100 customers in the  test dataset.

0.34 loss : a lower loss value  indicates  that model  predictions are close   to  actual  target. model  learn  undelaying pattern effectively.

early  stopping  :

stopping  training automatically bcz the validation  performance stopping  improvements. 
helping overfiting  , save  time  . 
"""
