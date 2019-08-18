
# coding: utf-8

# # Padding 
# ## August 18th 2019

# In[5]:


import tensorflow as tf 
from tensorflow import keras
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[21]:


sentence= ['I love my dog', 'I love my cat', 'You love my dog!', 'Do you think my dog is amazing?']

tokenizer= Tokenizer(num_words=100, oov_token= "<OOV>")
tokenizer.fit_on_texts(sentence)
word_index= tokenizer.word_index
print("Word index: \n",word_index, "\n")

sequences= tokenizer.texts_to_sequences(sentence)
print("Sequences: \n",sequences, "\n")

#Padding: 
padded= pad_sequences(sequences= sequences, padding= 'pre')
print("Padded sequences: \n", padded, "\n")

padded= pad_sequences(sequences= sequences, padding= 'post', truncating= 'post', maxlen=5)
print("Different way of padding: \n",padded)

