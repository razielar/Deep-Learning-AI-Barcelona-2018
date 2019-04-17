
# coding: utf-8

# # Image Generator
# ## April 17th 2019

# In[3]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Image Generator
# You can play with different sizes in 'target_size' without modifying your source code. 

# In[5]:


#### --- 1) Train:

train_datagen= ImageDataGenerator(rescale= 1./255) #normalization

#Put in the main directory that contain subdirectories that contain your data
train_generator= train_datagen.flow_from_directory(
    traindir,
    target_size= (300, 300), #Images can come in different shapes and we need all of them the same size to train NN
    batch_size= 128,
    class_mode= 'binary'
    
    )

#### --- 2) Test:

test_datgen= ImageDataGenerator(rescale= 1./255)

test_generator= test_datgen.flow_from_directory(
    testdir,
    target_size= (300, 300),
    batch_size= 128,
    class_mode= 'binary'

)

