"""

SGD            accuracy :     loss 
Momentum       accuracy :     loss 
RMSprop        accuracy :     loss 
Adam           accuracy :     loss 

best optimizer : 

acc high  loss low 

"""

# tensor flow : 
"""
1. lib ===> deep learning library , use build and train artificial neural network like ANN,CNN,RTSM etc

why we use ?? 
1. easy  build neural network
2. support  GPU and CPU 
3. Automatic backpropagation
4. production-ready


keras ??? 

===> keras high level API  inside tensorflow that  makes  building neural network easy 
and fast. 

ex : model = tf.keras.Sequential([
    tf.keras.layers.dense(16, activation='relu'),
    tf.keras.layers.dense(1, activation='sigmoid')
])

data set  : brest cancer dataset

features :  sample  569   features 30  classes 2   target : benign or malignant

1. import lib 
2. load data  
     x =features (30 columns), 
     y = target 
3. spilt :  teast size = 0.2  ====> 80 % train , 20 % test
4. feature scaler :   
    standard scaler 
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test) 

5. build model :
    model = tf.keras.Sequential([
    tf.keras.layers.dense(16, activation='relu'),input_shape=(30,),
    tf.keras.layers.dense(8, activation='relu'),
    tf.keras.layers.dense(1, activation='sigmoid')
    
    first  layer : 
    dense(16)   
    30 features  -----> 16 hidden neurons

    second layer :

    dense(8)
    16 hidden neurons  -----> 8 hidden neurons 

    output layer :
    dense(1)

6.compile model :
    optimizer="adam",  # fast , accurate  adaptive momentum estimation
    loss = "binary_crossentropy",
    merics=["accuracy"]
    
7. train model : 
    mode.fit(x_train,
    y_train,
    epochs=20,
    batch_size=32,  # to run cpu /GPU faster so divide in to small part , perform best
    validation_split=0.2,   ====> trainng 80 %  test :20% and test split
    verbose=1)

8. evaluate model :
    print(loss)
    print(accuracy)
    
9.predict : 
    mode.predict(x_test[ :5])
    
model summary : 

model : sequential

layer           output shape         param
dense            (None, 16)           496  ====>30 * 16  =480 +16  ===>496 
dense_1          (None, 8)            136  ====> 16 * 8 =128 +8   ===>136 
dense_2          (None, 1)            9    ====> 8 * 1 =8 +1     ===>9
  
30 fetures =====> 16 hidden neurons   ====> (input * nerons) + bias  ===>496 

2 layer : ====> 16  * 8  + 8   ====> 136
1 layer : ====> 8   * 1  + 1   ====> 9 

total param : 496 + 136 + 9 = 641
sequential :  layers are arranged one after another
input ---> features  ---> hidden layer---->Hidden layer  ----> output layer

device placement :  (CPU / GPU)
tf.config.list_physical_devices('GPU')

tensor processing unit 


"""


