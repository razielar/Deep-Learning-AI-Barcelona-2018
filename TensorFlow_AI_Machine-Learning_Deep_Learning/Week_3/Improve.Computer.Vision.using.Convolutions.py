
# coding: utf-8

# # Improving Computer Vision using Convolutions
# ## April 15th 2019

# In[24]:


import os, re, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ## Neural Networks

# First we are going to try only the Neural Network without Convolution and Max_Pooling to compare

# In[13]:


#Import the data:
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels)= mnist.load_data()
training_images= training_images/ 255.0
test_images= test_images/ 255.0

#1) Define the model:
model= tf.keras.models.Sequential([
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation= tf.nn.relu),
    tf.keras.layers.Dense(10, activation= tf.nn.softmax)
    
])
#2) Compile the model:
model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
#3) Train the model:
model.fit(training_images, training_labels, epochs= 5)
#4) Evaluate the model: 
test_loss= model.evaluate(test_images, test_labels)



# The accuracy on the train set was **0.89** and **0.86** in the test set. Not bad but we can improve the accuracy with adding Convolution layers before to the NN

# ## Convolutional Neural Network

# There's a bit of a change here in that the training data needed to be reshaped. That's because the first convolution expects a **single tensor** containing everything, so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1, and the same for the test images. If you don't do this, you'll get an error.

# In[23]:


print(tf.__version__)

#Import the data: 
mnist= tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#Modify the dimension of the training and test images: 
training_images= training_images.reshape(60000, 28, 28, 1) #Number of images, 28x28 pixels, 1 channel gray scale images
training_images= training_images/ 255.0
test_images= test_images.reshape(10000, 28, 28, 1)
test_images= test_images/ 255.0
#Create the model: 

model= tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', input_shape= (28, 28, 1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dense(10, activation= 'softmax')
    
])

model.compile(optimizer= 'adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs= 5)
test_loss= model.evaluate(test_images, test_labels)


# We see that the accuracy improved from **0.89** to **0.93**, and in the test from **0.86** to **0.90**

# ## Visualizing the Convolutions and Pooling

# In[73]:


tmp_test_images =test_images.reshape(10000, 28, 28)

plt.subplot(3, 1, 1)
plt.imshow(tmp_test_images[0])
plt.subplot(3, 1, 2)
plt.imshow(tmp_test_images[7])
plt.subplot(3, 1, 3)
plt.imshow(tmp_test_images[18])


# In[87]:


f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=18
CONVOLUTION_NUMBER = 2 #From 0 to 63, 64 convolutions

from tensorflow.keras import models
layer_outputs = [i.output for i in model.layers]
activation_model = tf.keras.models.Model(inputs= model.input, outputs= layer_outputs)
for x in range(0,4):
    f1= activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap= 'inferno')
    axarr[0, x].grid(False)
    f2= activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap= 'inferno')
    axarr[1,x].grid(False)
    f3= activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap= 'inferno')
    axarr[2,x].grid(False)


# ## Modify the CNN architecture

# In[88]:


import tensorflow as tf
print(tf.__version__)
#Import the data:
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

#Modify the model from 64 to 32, and only one Convolution and one Max_Pooling

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)


# We see that the accuracy improved from **0.93** to **0.99**, and in the test from **0.90** to **0.98**

# In[89]:


model.summary()

