import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset_path ="CNN" 

dataset =tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    label_mode='int',
    image_size=(128,128),
    batch_size=2,
    shuffle=True,
    seed =42
    )

# get class name :
class_names = dataset.class_names
print("class_names : ",class_names)

# display images :

plt.figure(figsize=(10,10))
for  images ,labels in dataset.take(1):
    for i in range(len(images)):
        plt.subplot(1,len(images),i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
# plt.show()

# convert dataset to numpy array :
images =[] 
labels =[] 

for batch_images ,batch_labels in dataset:
    images.extend(batch_images.numpy())
    labels.extend(batch_labels.numpy())
    
images=np.array(images)
labels=np.array(labels)
print("images.shape : ",images.shape)
print("labels.shape : ",labels.shape)

# split dataset into train and test set :
X_train,X_test,y_train,y_test=train_test_split(images,
                                               labels,
                                               test_size=0.2,
                                               random_state=42,
                                               stratify=labels
                                               )

print("X_train.shape : ",X_train.shape)
print("X_test.shape : ",X_test.shape)  
# X_train.shape :  (8, 128, 128, 3)
# X_test.shape :  (2, 128, 128, 3)

# normalize dataset :
"""
original image  : 0 to 255 
after normalization : 0 to 1
"""
X_train=X_train.astype("float32")/255.0 
X_test=X_test.astype("float32")/255.0

# create CNN  model :  
model = tf.keras.Sequential([
    # first convolution  layer :
    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),
    # batch normalization layer :
    tf.keras.layers.BatchNormalization(),
    # max pooling layer :
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # second convolution  layer :
    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # third convolution  layer :
    tf.keras.layers.Conv2D(
        128,
        (3,3),
        activation="relu"
        ),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # flattern :
    tf.keras.layers.Flatten(),
    
    # dense layer :
    tf.keras.layers.Dense(128,activation="relu"),
    
    # dropout layer :
    tf.keras.layers.Dropout(0.5),
    # output  ----->layer ----> only 2 classes   circle and square
    
    tf.keras.layers.Dense(1,activation="sigmoid")
])

# summary  : 
model.summary()

# compile model :
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# train model :
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=2,
    validation_data=(X_test,y_test)
)

# evaluate model :
test_loss,test_acc=model.evaluate(X_test,y_test)
print("test_loss : ",test_loss)
print("test_acc : ",test_acc)

#plot training and validation accuracy ,plot training and validation loss : 

# predict : 
image =X_test[0]   # -----> select first image
image_for_prediction=image.reshape(1,128,128,3)

prediction  = model.predict(image_for_prediction,verbose=0)

probability=prediction[0][0]
print("probability : ",probability)

# convert probability to class name :

if probability>=0.5:
    predicted_class = class_names[1]

else :
     predicted_class = class_names[0]
    
# actual class name :

actual_class =class_names[y_test[0]]
print("predicted_class : ",predicted_class)
print("actual_class : ",actual_class)

if predicted_class==actual_class:
    print("Correct prediction")
else :
    print("Wrong prediction")
    
# display  +prediction :
plt.figure(figsize=(10,10))
plt.imshow(X_test[0])
plt.title(
    f"predicted class : {predicted_class} , actual class : {actual_class}"
)
plt.axis("off")
plt.show()

# predict  on test set :

"""output  : 
test_loss :  3.192456007003784
test_acc :  0.5
probability :  0.9983131
predicted_class :  Circle
actual_class :  circle
Wrong prediction"""