import numpy as np
import tensorflow as tf
  # 4,2 
X=np.array([
    [2,3],
    [4,5],
    [6,7],
    [8,9]
])
y=np.array([
    [0],
    [0],
    [1],
    [1]
])
# network  :

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

"""# build model :
model.build(input_shape=(None,2))

#summary :
model.summary()
"""
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='binary_crossentropy', 
    metrics=['accuracy'])

model.fit(X,y,epochs=100,verbose=1)

# prediction  : 
predictions = model.predict(X)

for i , value in enumerate(predictions) :
    print(f"sample {i+1} prediction : {value[0]:.2f}")


