"""import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

df = pd.DataFrame({
    "study_hrs" :[2,3,4,5,6,7,8,9], 
    "marks" : [34,40,56,66,72,83,91,98]
})

x = df["study_hrs"].values   # [2,3,4,5,6,7,8,9]

x = (x - np.mean(x)) / np.std(x)  # std .scaler  

# sigmoid  : 1/ 1 + e^-z 
# tanh     : e^z - e^-z / e^z + e^-z 
# Relu     : max(0,z)

sigmoid = 1 / (1 + np.exp(x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.plot(df["study_hrs"], sigmoid, label="Sigmoid")
plt.plot(df["study_hrs"], tanh, label="Tanh")
plt.plot(df["study_hrs"], relu, label="ReLU")
plt.title("Activation Functions on Student Hours Dataset")
plt.xlabel("Study Hours")
plt.ylabel("Activation Output")
plt.legend()
plt.grid(True)
plt.show()

"""
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
