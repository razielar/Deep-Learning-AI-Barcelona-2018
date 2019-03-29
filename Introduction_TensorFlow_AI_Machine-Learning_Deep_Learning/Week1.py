
# coding: utf-8

# # Neural Network
# ## March 29th 2019
# ### Using Keras

# # 1) A primer in Machine Learning

# In[5]:


### Libraries: 

import keras
from keras.models import Sequential
import numpy as np


# The simplest NN: the one that only has one neuron on it <br>
# In keras **dense**= define a layer of connected neurons

# In[3]:


# The simplest NN: one neuron

model= Sequential([keras.layers.Dense(units= 1, input_shape=[1])])


# Optimizer= **sgd**= stochastic gradient descent

# In[7]:


model.compile(optimizer= keras.optimizers.sgd() , loss= keras.losses.mean_squared_error )


# In[10]:


### Data: 

xs= np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype= float)
ys= np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype= float)


# In[13]:


model.fit(xs, ys, epochs= 500)


# In[14]:


print(model.predict([10.0]))


# Our expected value should be **19**, because the formula is **Y= 2X-1** <br>
# **One possible** explanation is that there's very few data only six points and there's no gurantee that **there's a linear relation** of Y and X. <br>
# **Second** because NN deals in **probability**
