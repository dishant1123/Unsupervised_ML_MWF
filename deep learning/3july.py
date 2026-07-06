"""
what is  deep learning ?? 
 ===> subset of  machine learning 
 ===> use  neural network
 ===> complex pattern to learn  from  large data 
 
deep learning is a techniques that teaches computers to learn complex pattern from data using nerual networks with multiple layers. 

que : why we use deep larning we  also already  use  unsupervised learing  for complex pattern ?

ex :cat vs dog classification  ===> 20000 images for  cat and dog 
    traditional ML  : 
        features extract : ear shape ,nose shape , tail shape,color, teeth ,eye size 
    
    image  
     |
    features extract
     |
    ML Model 
     |
    prediction  

deep learning : neural network automatically detect the  image  of  ear ,eye,nose,tail 

    image 
     | 
    deep neural network
     |
    prediction  

diff between  ML  VS Deep learning
                ML                            DL 
1. subset of AI                        subset of ML 
2. features extraction  manual         features extraction automatic
3. low  GPU                                   GPU useful  
4. stracture  data                      image,text,audio,video 

ex: ML 

house  price prediction
input  :area , bedroom , bathroom , price ====> model  predict price 

ex : DL :face recognition
    eye  distance  ,nose width ,face  length   ===> predict face 
    
why  we use  deep learning ?? 
1.large  amount  of data  . ===> 20000 images for  cat and dog,voice recording 
2.powerful  GPU ===> GPU useful
3.better algorithm ===> 
    1. CNN : convolutional neural network
    2. RNN : recurrent neural network
    3. ANN : artificial neural network
    4. LSTM : long short term memory
    5. transformer

1.biological neuron
    humna brain contains appox 82 million neurons
    
    input  signal 
        | 
    dendrite 
        |
    cell body 
        |
    axon 
        | 
    output signal
 
2.artificial neuron : 
    inspired by biological neuron
    
    input 
     |
    weight
     |
    weight sum 
     | 
    activation function  
     |
    output  

y = f(w1x1 + w2x2  + b)
 
 x =input  
 w= weight 
 b = bias
 f= activation function
 
flow  : 

collect  data  ===>data  preprocessing ===>build model ==>train model ===> evaluate model ===>deploy model
"""
