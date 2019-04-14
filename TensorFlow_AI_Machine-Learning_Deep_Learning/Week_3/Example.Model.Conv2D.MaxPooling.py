
# coding: utf-8

# # Convolution and Pooling layers
# ## April 14th 2019

# In[3]:


import numpy as np
import tensorflow as tf


# In[10]:


#We're going to define a model with 2 convolution layer, and then one input layer, one hidden layer and 
#one output layer
#Sequential: defines a SEQUENCE of layers in the NN
    #First convolution 2D layer with 64 filters and 3x3 filter dimension; Filters are need for the convolution
    #Max_Pooling take the maximum value: compressing, this will perserve the features that were highlighted by the 
    #convolution
#Neural Network with a input layer (Flatten), 
    #one hidden layer (128 neurons), 
    #output layer (10 neurons) 10 catogories
#Flatten: turns 28X28 (two dimensions) into a linear array 784X1
#Dense: fully connected neurons
    #Activation function in the 128 neurons: ReLU
    #Softmax: take the biggest value
    

model= tf.keras.models.Sequential([
    #First Convolution
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    #Second Convolution
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    #Neural Network: 
    tf.keras.layers.Flatten(), #Flat the 28x28 into one dimension array
    tf.keras.layers.Dense(128, activation= tf.nn.relu), #One hidden layer with 128 neurons
    tf.keras.layers.Dense(10, activation= tf.nn.softmax) #Output layer fully connected 'Dense' with 10 neurons 
])


# In[11]:


model.summary()


# **Output shape:** <br><br>
# **1)**  **First Convolution:** 26, 26; the filter is a 3x3 filter. The edges of the pixels don't have any neighbors. This means 2 pixels less in X and Y. <br>
# **2) Max Pooling:** (2,2)= it get reduce from 26x26 to 13x13. <br>
# **3) Second Convolution:** we lose one pixel margin as before there 13-2; 11. <br>
# **4) Max Pooling:** 11/2 5.5 => 5. <br>
# **5) Input layer:** 5x5 input images, due the convolution there're 64 new images with 5x5= 25; 25x64= 1600. <br>
# **6) Hidden layer:** 128 neurons. <br>
# **7) Output layer:** 10 neurons. 
# 
# 
# 
# 
# 
