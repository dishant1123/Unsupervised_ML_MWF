import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.array([
    [1, 2, 3, 4],
    [2, 3, 4, 5],
    [3, 4, 5, 6],
    [4, 5, 6, 7],
    [5, 6, 7, 8],
    [6, 7, 8, 9],
    [7, 8, 9, 10]
], dtype=float)

y = np.array([
    5,
    6,
    7,
    8,
    9,
    10,
    11
], dtype=float)

# 2. Reshape data for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))
print("X shape:", X.shape)
print("y shape:", y.shape)

# 3. Create RNN model
model = Sequential([
    SimpleRNN(10, activation="tanh", input_shape=(4, 1)),
    Dense(1)
])
# 4. Compile model
model.compile(
    optimizer="adam",
    loss="mse"
)
# 5. Train model
model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)
print("Training completed!")

# 6. Test the model
test_data = np.array([
    [8, 9, 10, 11]
], dtype=float)

test_data = test_data.reshape((1, 4, 1))
prediction = model.predict(test_data, verbose=0)
print("Input:", [8, 9, 10, 11])
print("Predicted next number:", prediction[0][0])