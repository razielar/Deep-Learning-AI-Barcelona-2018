#!/usr/bin/env python
# coding: utf-8

# # Word-based encoders
# ## August 18th 2019

# In[4]:


#Import modules:

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# In[10]:


setence= ['I love my dog', 'I love my cat']

tokenizer= Tokenizer(num_words=100)
tokenizer.fit_on_texts(setence)
word_index= tokenizer.word_index


# In[11]:


#Key is the word and value the word token
#Tokenizer convert upper to lower-case 
word_index


# In[17]:


sentence= ['I, love my dog', 'i love my cat', 'You love my dog!']
tokenizer= Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentence)
word_index= tokenizer.word_index


# In[18]:


#Uses the same token for dog and dog!
word_index


# In[ ]:





# In[ ]:




