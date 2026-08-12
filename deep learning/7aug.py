"""
hyperarameter : is value that  you choose before training the model. 

ex: 
----->numbers of hidden layers
----->number of neurons in each layer
----->learning rate,batch_size,optimizer,weight initialization
"""

import numpy as np
import tensorflow as tf
from tensorflow .keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.optimizers import Adam,SGD

data =load_breast_cancer()

X=data.data
y=data.target

# spilt : 
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0)

# features scaling :
scaler = StandardScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)

# base model  :

'''model = Sequential()

model.add(Dense(16,activation='relu',input_shape=(30,)))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
'''
# hyperparameter  

layers =3
neurons =32 
batch_size =16 
epochs =30 

learning_rate =0.001

optimizer =Adam(learning_rate)
initializer ='he_normal'   # kernel_initializer='he_normal'

model = Sequential()

model.add(Dense(neurons,activation='relu',input_shape=(30,)))
model.add(Dense(neurons,activation='relu',kernel_initializer=initializer))
model.add(Dense(neurons,activation='relu',kernel_initializer=initializer))

model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2
)

loss ,acc = model.evaluate(X_test, y_test)
print(' accuracy:', acc)
print(' loss:', loss)

"""
hyperparameter : 

model A :  model = Sequential([
    Dense(16, activation='relu', input_shape=(30,)),
    dense(8, activation='relu'),
    dense(1, activation='sigmoid')   =====> only  hidden layer
])

model B :  model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu' ),
    dense(8, activation='relu'),
    dense(1, activation='sigmoid')   ====> 3 hidden layers
])

2. number  of neurons in each layer : dense 8  dense 16 
3. batch size : 
    16 ----> slow but accurate update 
    32 ----> balance 
    64 ----> fast but may be accuracy reduce 
    128 ---->very fast for large dataset
    
4. epochs :
5.optimizer : adam 

"""
# automated hyperparameter tuning KerasTuner

# pip install keras-tuner 
'''
import keras_tuner as kt

# build fuction  :


def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int("units", min_value=16, max_value=128, step=16), activation="relu", input_shape=(30,)))

    # Tune neurons
    model.add(
        Dense(
            units=hp.Int("units", min_value=16, max_value=128, step=16),
            activation="relu"
        )
    )

    model.add(Dense(1, activation="sigmoid"))

    # Tune optimizer
    optimizer = hp.Choice(
        "optimizer",
        values=["adam", "rmsprop", "sgd"]
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

tuner =kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='my_tuner',
    project_name='breast_cancer'
)

tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

best_params =tuner.get_best_hyperparameters()[0]
print(best_params.get('units'))

# evaluate best model :
loss ,acc = model.evaluate(X_test, y_test)
print(' accuracy:', acc)
print(' loss:', loss)'''


"""
neural network  -----> connection  between  weight 
before training the model  ----> intial value  of weight 

kernel_initializer ----> tells keras how to  initialize those weights
kernel_initializer specifies the method  used to intialize the weights of layer before training the model. 

"""