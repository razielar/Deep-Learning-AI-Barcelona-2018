
# coding: utf-8

# # Experiment with filters and pools 
# ## April 17th 2019

# In[57]:


import numpy as np
import cv2
from scipy import misc
import matplotlib.pyplot as plt


# ## See and example image
# We are going to explore how convolutions work on a 2D gray scale. <br> 
# Let's use the 'ascent' image from scipy.

# In[58]:


### Load the image
i = misc.ascent()

### Plot the image
plt.grid(False)
plt.gray()
plt.imshow(i)
plt.show()


# ## Create a filter and apply a Convolution
# We are going to create a filter as a 3x3 array <br>
# Let's create a convolution that only passes through sharp edges and lines. <br>
# If all the digits in the filter don't add up to 0 or 1, you should probably do a weight to get it to do so.
# So, for example, if your weights are 1,1,1 1,2,1 1,1,1. They add up to 10, so you would set a **weight of .1** if you want to normalize them weight  = 1 <br>
# ## Convolution
# We will iterate over the image, leaving a 1 pixel margin, and multiply out each of the neighbors of the current pixel by the value defined in the filter. <br>
# i.e. the current pixel's neighbor above it and to the left will be multiplied by the top left item in the filter etc. etc. We'll then multiply the result by the weight, and then ensure the result is in the range 0-255

# In[62]:


### Copy the original image:

i_transformed = np.copy(i) #Copy the Numpy array
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

### --- 1) This filter detects edges nicely

### Experiment with different filters
# 3x3 array:
#filter = [[-1, -2, -1], [0, 0, 0], [1,2,1]] #vertical edges
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] #horizontal edges
weight= 1

### --- 2) Convolution

for x in range(1, size_x -1):
    for y in range(1, size_y -1):
        convolution= 0.0
        convolution = convolution + (i[x - 1, y-1] * filter[0][0])
        convolution = convolution + (i[x, y-1] * filter[0][1])
        convolution = convolution + (i[x + 1, y-1] * filter[0][2])
        convolution = convolution + (i[x-1, y] * filter[1][0])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x+1, y] * filter[1][2])
        convolution = convolution + (i[x-1, y+1] * filter[2][0])
        convolution = convolution + (i[x, y+1] * filter[2][1])
        convolution = convolution + (i[x+1, y+1] * filter[2][2])
        convolution = convolution * weight
        
        if (convolution < 0):
            convolution = 0
        if (convolution > 255):
            convolution= 255
            
        i_transformed[x, y] = convolution
        


# ## Let's see the effect of the convolution

# In[63]:


plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()


# ## Max Pooling
# We're going to do a (2,2) max pooling. Thus the **new image** will be **1/4 the size of the old**. We'll see that **the features get maintained despite this compression**

# In[64]:


new_x= int(size_x/2)
new_y= int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels= []
        pixels.append(i_transformed[x,y])
        pixels.append(i_transformed[x+1, y])
        pixels.append(i_transformed[x, y+1])
        pixels.append(i_transformed[x+1,y+1])
        newImage[int(x/2), int(y/2)] = max(pixels)

### --- Plot the new image

plt.gray()
plt.grid(False)
plt.imshow(newImage)
plt.show()

