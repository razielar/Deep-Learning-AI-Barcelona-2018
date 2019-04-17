
# coding: utf-8

# # Defining a ConvNet to use complex images
# ## April 17th 2019

# In[2]:


import tensorflow as tf


# Output layer of the NN, we only have one neuron for two classes, but it's because we have another activation function. **Sigmoid** is great for **binary classification**. We can use **two neurons** for the **sofmax function**, but it's *more efficient* using *one neuron using sigmoid function*. 

# In[5]:


model= tf.keras.models.Sequential([
    
    ### Three Convolution layers
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu',
                          input_shape= (300, 300, 3)), #Color images
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    ### Neural Network:
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(1, activation= 'sigmoid') #One neuron for two classes
])

model.summary()


# ## Description of the model:
# 
# ### Convolution layer:
# **First Convolution:** 300x300 images become 298x298; lossing the pixels from the edges. <br>
# **Max Pooling:** 1/4 of the image because of the (2,2) <br>
# **Second Convolution:** 147x147 images; lossing the pixels from the edges. <br>
# **Max Pooling:** 1/4 of the image because of the (2,2) <br>
# **Third Convolution:** 71x71 images; lossing the pixels from the edges. <br>
# **Max Pooling:** 1/4 of the image because of the (2,2) <br>
# 
# ### Neural Network:
# 
# **Input layer:** Flatten the 35x35 images with 64 convolutions <br>
# **Hidden layer:** 512 neurons <br>
# **Output layer:** one neuron using the sigmoid function for binary classification
# 

# In[7]:


35*35*64

