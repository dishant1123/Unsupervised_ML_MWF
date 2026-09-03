# RNN ,LSTM,GRU : 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_absolute_error,mean_squared_error 
import tensorflow  as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN,LSTM,GRU,Dense

# random seed :

np.random.seed(42)
tf.random.set_seed(42)

# load data :

data ={
    "days" : list(range(1,26)),
    "temperature" :[
        30.0,30.5,31.0 ,31.8 ,32.2 ,
        33.0 ,33.5 ,34.0 ,34.5 ,35.0 ,
        35.5 ,36.0 ,36.5 ,37.0 ,37.5 ,
        38.0 ,38.5 ,39.0 ,39.5 ,40.0 ,
        40.5 ,41.0 ,41.5 ,42.0 ,42.5 
        
    ]
}
df = pd.DataFrame(data)
print(df)

# visualize data :

plt.figure(figsize=(10,5))
plt.plot(df["days"],
         df["temperature"],
         marker="o")
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.grid(True)
plt.show() 

# extract temperature : 

temperature =df['temperature'].values
temperature = temperature.reshape(-1,1)
print(temperature)

# normalization  : 

scaler = MinMaxScaler()
scaled_temperature = scaler.fit_transform(temperature)
print(scaled_temperature[ :10])# create  sequences : 

sequence_lenght =5 
x=[] 
y=[] 

for i in range(len(scaled_temperature) - sequence_lenght):
    x.append(scaled_temperature[i:i+sequence_lenght])
    y.append(scaled_temperature[i+sequence_lenght])
    
x=np.array(x)
y=np.array(y)

print(x)
print(y)

# check  shape : 
print("x shape : ",x.shape)
print("y shape : ",y.shape)

# split  : 
train_size = int(0.8*len(x))
x_train,x_test = x[:train_size],x[train_size:]
y_train,y_test = y[:train_size],y[train_size:]

print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

# rnn model:
print("****RNN****\n")

rnn_model = Sequential([
    SimpleRNN(32,
              input_shape=(sequence_lenght,1)),
    Dense(1)
])
# complie : 
rnn_model.compile(
    optimizer ="adam",
    loss ="mse"
)
# summary  : 

rnn_model.summary() 

# train : 
rnn_history = rnn_model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=1
)

#LSTM model : 
print("****LSTM****\n")

lstm_model = Sequential([
    LSTM(32,
              input_shape=(sequence_lenght,1)),
    Dense(1)
])
# complie : 
lstm_model.compile(
    optimizer ="adam",
    loss ="mse"
)
# summary  : 

lstm_model.summary() 

# train : 
lstm_history = lstm_model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=1
)

# GRU : 
print("****GRU****\n")
gru_model = Sequential([
    GRU(32,
              input_shape=(sequence_lenght,1)),
    Dense(1)
])
# complie : 
gru_model.compile(
    optimizer ="adam",
    loss ="mse"
)
# summary  : 

gru_model.summary() 

# train : 
gru_history = gru_model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=1
)
# prediction  : 

print("*********prediction *************\n")

rnn_predict = rnn_model.predict(x_test)
lstm_predict = lstm_model.predict(x_test)
gru_predict = gru_model.predict(x_test)

# convert  prediction to original scale : 
y_test_actual =scaler.inverse_transform(y_test)
rnn_predict_actual = scaler.inverse_transform(rnn_predict)
lstm_predict_actual = scaler.inverse_transform(lstm_predict)
gru_predict_actual = scaler.inverse_transform(gru_predict)

# result : 

results = pd.DataFrame({
    "Actual Temperature" : y_test_actual.flatten(),
    "RNN Predicted" : rnn_predict_actual.flatten(),
    "LSTM Predicted" : lstm_predict_actual.flatten(),
    "GRU Predicted" : gru_predict_actual.flatten()
})
print(results)

# calculate  the  loss using  MAE , MSE 
# model comparsion 
# choose  best  model  : 

"""
compare.loc[compare['MAE'].idxmin()]

print("best model")
"""

# plot actual vs  rnn : 

plt.figure(figsize=(10,5))
plt.plot(
    y_test_actual ,
    marker ='o',
    label ="actual"
)
plt.plot(
    rnn_predict_actual,
    marker ='x',
    label ="rnn"
)
plt.xlabel("test sample")
plt.ylabel("temperature")
plt.legend()
plt.grid(True)
plt.show()

# plot actual vs  lstm :

plt.figure(figsize=(10,5))
plt.plot(
    y_test_actual ,
    marker ='o',
    label ="actual"
)
plt.plot(
    lstm_predict_actual,
    marker ='x',
    label ="LSTM"
)
plt.xlabel("test sample")
plt.ylabel("temperature")
plt.legend()
plt.grid(True)
plt.show()

# plot actual vs  GRU :

plt.figure(figsize=(10,5))
plt.plot(
    y_test_actual ,
    marker ='o',
    label ="actual"
)
plt.plot(
    gru_predict_actual,
    marker ='x',
    label ="GRU"
)
plt.xlabel("test sample")
plt.ylabel("temperature")
plt.legend()
plt.grid(True)
plt.show()



