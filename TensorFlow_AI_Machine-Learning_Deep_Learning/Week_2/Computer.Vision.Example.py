
# coding: utf-8

# # Computer Vision Example
# ## April 6th 2019
# ### Train a CNN to recognize 10 categories of clothes from the fashion MNIST dataset

# In[6]:


import tensorflow as tf 
import matplotlib.pyplot as plt
print(tf.__version__)


# ## Load and analyze the MNIST dataset
# Fashon MNIST: <br>
# 70 K images <br>
# 10 categories <br>
# 28X28 pixels in gray scale

# In[5]:


#Load the MNIST dataset:
mnist= tf.keras.datasets.fashion_mnist

#Train and test: 
(training_images, training_labels), (test_images, test_labels)= mnist.load_data()


# ### How the data looks like

# In[15]:


plt.imshow(training_images[0])
print("Training_images:", training_images[0], "\n")
print("Shape Training_images:", training_images.shape, "\n")
print("Training_labels:", training_labels[0], "\n")
print("Shape Training_labels:", training_labels.shape, "\n")


# ### Normalize the data
# An image has values from 0-255 

# In[17]:


training_images= training_images/ 255.0
test_images= test_images / 255.0


# ## Build the model

# In[21]:


# 1) DEFINE THE MODEL:
#Neural Network with a input layer (Flatten), 
    #one hidden layer (128 neurons), 
    #output layer (10 neurons) 10 catogories
#Sequential: defines a SEQUENCE of layers in the NN
#Flatten: turns 28X28 (two dimensions) into a linear array 784X1
#Dense: fully connected neurons
    #Activation function in the 128 neurons: ReLU
    #Softmax: take the biggest value

model= tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(128, activation= tf.nn.relu),
                                  tf.keras.layers.Dense(10, activation= tf.nn.softmax)])

# 2) COMPILE:
model.compile(optimizer= tf.train.AdamOptimizer(),
             loss= 'sparse_categorical_crossentropy',
             metrics= ['accuracy'])
# 3) FIT
model.fit(training_images, training_labels, epochs= 5)


# In[22]:


# 4) PREDICT
model.evaluate(test_images, test_labels)


# # Exploration Exercises

# In[27]:


classification= model.predict(test_images)
#It's the probability that this item is each of the 10 classes
print("Classification:",classification[0], "\n")
print("Test_label:",test_labels[0], "\n")


# In[32]:


print("We know that the first element is an ankle boot bc is 9th element (last one)", "\n")
classification[0]


# ## Increase the number of Neurons in the hidden layer

# In[34]:


import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
             metrics= ['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])


# It increase the accuracy from **87.61** (128) to **98.17** (1024)

# In[35]:


28*28


# ## Increase the number of hidden layers

# In[36]:


import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

#Increase the number of hidden layers:
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
             metrics= ['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])


# There isn't any significant impact from **98.17** to **98.18** of increasing a hidden layer because this is a simple dataset. For far more complex data (including color images to be classified as flowers), extra layers are often necessary

# ## Classify the images without Normalization

# In[37]:


import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#training_images=training_images/255.0
#test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])


# ## Callback

# Earlier when you trained for extra epochs (**overfitting**). It might have taken a bit of time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the training when I reach a desired value?' -- i.e. **95% accuracy** might be enough for you, and if you reach that after 3 epochs, why sit around waiting for it to finish a lot more epochs. For that let's try **Callback**

# In[49]:


import tensorflow as tf
print(tf.__version__)

#if (logs.get('loss') < 0.05):
#print("\nReached a loss lower than 0.05 so cancelling training!\n")
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')> 0.95):
            print("\nReached an accuracy higher than 95% so cancelling training\n")
            self.model.stop_training= True

callbacks = myCallback()

mnist = tf.keras.datasets.mnist

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
             metrics= ['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks= [callbacks])

