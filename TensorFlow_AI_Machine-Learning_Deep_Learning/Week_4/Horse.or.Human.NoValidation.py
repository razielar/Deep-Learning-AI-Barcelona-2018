
# coding: utf-8

# # Horse or Human No-validation 
# ## Only using training test
# ### April 21st 2019

# In[92]:


import tensorflow as tf
#Data Generator:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import os
import zipfile
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# ## Download and Process the images

# In[9]:


### --- Download the Horse & Human dataset:

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip     -O /tmp/horse-or-human.zip #/tmp folder')


# ### Image Generator: 
# We **do not explicitly label the images** as *horses* or *humans*.
# In previous examples we had labels from the images. The Image Generator is coded to read images from **subdirectories**, and **automatically label them from the name of the subdirectory.** 

# In[35]:


#Location of the dataset:
local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
#Extract the content of the folder
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()

### --- Directory with our training horse pictures: 
train_horse_dir= os.path.join('/tmp/horse-or-human/horses/')

### --- Directory with our training human pictures: 
train_human_dir= os.path.join('/tmp/horse-or-human/humans/')

train_horse_name= os.listdir(train_horse_dir)
train_human_name= os.listdir(train_human_dir)

#Print the last 10 images of each train label: horses and humans
print("Horses train images:",train_horse_name[:10], "\n")
print("Humans train images:",train_human_name[:10], "\n")

#Print the total number of training horse and human images:
print("Total training horse images:", len(train_horse_name), "\n")
print("Total training human images:", len(train_human_name), "\n")



# In[65]:


# Parameters for our graph:
nrows= 4; ncols= 4
#Index for iterating over images:
pic_index= 0
# Set up matplotlib fig, and size it to fit 4x4 pics

### --- If you don't use the two lines the images are to small:
fig= plt.gcf() #Get a reference to the current figure
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8

next_horse_pic= [os.path.join(train_horse_dir, i) 
                 for i in train_horse_name[pic_index-8:pic_index]]

next_human_pic= [os.path.join(train_human_dir, i)
                for i in train_human_name[pic_index-8:pic_index]]

#next_horse_pic+next_human_pic= 16, i=range(0,17) => 16 elements
for i, image_path in enumerate(next_horse_pic+next_human_pic):
    sp= plt.subplot(nrows, ncols, i+1)#0+1=1; 15+1=16
    #sp.axis('Off') #get rid of the axes
    
    img= mpimg.imread(image_path)
    plt.imshow(img)

plt.show()
    


# ## Preprocessing

# ### Image Generator
# Our data generator will read pictures in our source folders, convert them to float32 tensors, and feed them (with their labels) to our network.

# In[72]:


# All images will be rescaled by 1./255:
train_datgen= ImageDataGenerator(rescale= 1./255)

# Flow training images in batches of 128 using train_datgen

train_generator= train_datgen.flow_from_directory(
    '/tmp/horse-or-human/',#Source directory
    target_size= (300, 300), # 300x300 images
    batch_size= 128, #load images in batches
    class_mode= 'binary' #horse & human classification
    )


# ## ConvNet

# Note that output layer of the NN we will use a **sigmoid activation** so our output will be a *single scalar* **between 0 and 1**, encoding the probability that the current image is class 1 (as opposed to class 0).

# In[77]:


### --- 1) Define the model:

model= tf.keras.models.Sequential([
    
    #First convolution
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu',
                          input_shape= (300, 300, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    #Second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #Third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #Fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #Fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    
    #NN
    #Input layer: Flatten:
    tf.keras.layers.Flatten(),
    #512 neurons
    tf.keras.layers.Dense(512, activation= 'relu'),
    #Output layer:
    tf.keras.layers.Dense(1, activation= 'sigmoid')
    
])

model.summary()


# Using **RMSprop** is preferable to **stochastic gradient descent** because RMSprop automates learning-rate tuning for us. 
# (Other optimizers, such as **Adam** and **Adagrad**, also automatically adapt the learning rate during training, and would work equally well here)

# In[87]:


print("Number of total images:", 1027, "\n")
print("Number of batches in Image Generator", 128, "\n")
print("Therefore: 1027 / 128 is", round(1027/128, 2))


# In[88]:


### --- 2) Compile the model:

model.compile(loss= 'binary_crossentropy',
             optimizer= RMSprop(lr=0.001),
             metrics= ['accuracy'])

### --- 3) Train the model: 15 epochs

history= model.fit_generator(
    train_generator, #Data Image Generator object
    steps_per_epoch=8, #to load all the batches
    epochs=15, #15 epochs
    verbose=1)


# ## Select a picture and run the model
# We are going to select a picture from [pixabay](https://pixabay.com/), and find [horses](https://pixabay.com/images/search/horse/)

# In[95]:


#import numpy as np
#from google.colab import files
#from keras.preprocessing import image

#uploaded = files.upload()

#for fn in uploaded.keys():
 
  # predicting images
  #path = '/content/' + fn
  #img = image.load_img(path, target_size=(300, 300))
  #x = image.img_to_array(img)
  #x = np.expand_dims(x, axis=0)

  #images = np.vstack([x])
  #classes = model.predict(images, batch_size=10)
  #print(classes[0])
  #if classes[0]>0.5:
    #print(fn + " is a human")
  #else:
    #print(fn + " is a horse")

