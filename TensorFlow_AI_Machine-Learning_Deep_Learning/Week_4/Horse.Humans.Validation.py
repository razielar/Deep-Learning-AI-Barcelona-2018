
# coding: utf-8

# # Horse and Humans Validation
# ## Use train and test 
# ### April 22nd 2019

# In[46]:


### Libraries:
import os
import zipfile
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Download and Preprocessing

# In[2]:


### --- 1) Download the training dataset

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip     -O /tmp/horse-or-human.zip')


# In[3]:


### --- 2) Download the validation (train) dataset

get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip     -O /tmp/validation-horse-or-human.zip')


# In[31]:


### Extract all the training dataset:
local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
### Extract all the validation dataset:
local_zip = '/tmp/validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()

#### --- Directory setup:

### Train:
train_horse_dir= os.path.join('/tmp/horse-or-human/horses')
train_human_dir= os.path.join('/tmp/horse-or-human/humans')

### Validation:
validation_horse_dir= os.path.join('/tmp/validation-horse-or-human/horses')
validation_humn_dir= os.path.join('/tmp/validation-horse-or-human/humans')

##### --- Print the files
train_horse_names= os.listdir(train_horse_dir)
train_human_names= os.listdir(train_human_dir)
validation_horse_names= os.listdir(validation_horse_dir)
validation_human_names= os.listdir(validation_humn_dir)

print("Train Horses: ",train_horse_names[:10], "\n")
print("Train Humans:", train_human_names[:10], "\n")
print("Validation Horses:", validation_horse_names[:10], "\n")
print("Validation Humans:", validation_human_names[:10], "\n")

#### --- Print the lenght of Train & Validation:

print("Total training horse images: ",len(train_horse_names), "\n")
print("Total training human images: ", len(train_human_names), "\n")
print("Total validation horse images: ", len(validation_horse_names), "\n")
print("Total validation human images:", len(validation_human_names), "\n")



# In[43]:


nrows= 4; ncols= 4

pic_index= 0

fig= plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, i)
                 for i in train_horse_names[pic_index-8: pic_index]]

next_human_pix = [os.path.join(train_human_dir, i)
                 for i in train_human_names[pic_index-8:pic_index]]

for i, j in enumerate(next_horse_pix+next_human_pix):
    #Set up subplots; subplot indices start with 1
    sp= plt.subplot(nrows, ncols, i+1)
    
    img= mpimg.imread(j)
    plt.imshow(img)
    
plt.show()


# ## Data Preprocessing

# ### Image Data Generator
# We will have one generator for the training images and one for the validation images. Our generators will yield batches of images of size 300x300 and their labels (binary). These generator can be then used with the Keras model methods that accept data generators as inputs: **fit_generator, evaluate_generator, predict_generator**

# In[53]:


### All images will be rescaled by 1./255
train_datagen= ImageDataGenerator(rescale= 1./255)
validation_datagen= ImageDataGenerator(rescale= 1./255)

### Train Data Generator:
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size= (300, 300),
    batch_size= 128,
    class_mode= 'binary'
)

### Validation Data Generator:
validation_generator= validation_datagen.flow_from_directory(
    '/tmp/validation-horse-or-human/',
    target_size= (300, 300),
    batch_size= 32,
    class_mode= 'binary'

)


# ## ConvNet

# Note the input shape is the desired size of the image 300x300 with 3 bytes color. <br>
# Neuronal Network: Output layer: It will contain a value from 0-1 where **0** for **1 class ('horses')** and **1** for the other **('humans')**. 

# In[56]:


### --- 1) Define the model:

model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    ### --- Neural Network:
    # Input layer Flatten:
    tf.keras.layers.Flatten(),
    # Hidden Layer: 512 neurons:
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer: 1 neuron. 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


# In[65]:


print("Number of total images:", 1027)
print("Number of batches in Image Generator", 128)
print("Therefore: 1027 / 128 is", round(1027/128, 2), "\n")

print("Number of total validation images:", 256)
print("Number of batches in Validation Generator:", 32)
print("Therefore: 256 / 32 is", round(256/32, 2))


# In[66]:


### --- 2) Compile the model:

model.compile(loss= 'binary_crossentropy',
             optimizer= RMSprop(lr=0.001),
             metrics= ['accuracy'])

### --- 3) Train the model: 15 epochs

history= model.fit_generator(
    train_generator, #Data Image Generator object
    steps_per_epoch=8, #to load all the batches
    epochs=15, #15 epochs
    verbose=1,
    validation_data= validation_generator,
    validation_steps= 8)

