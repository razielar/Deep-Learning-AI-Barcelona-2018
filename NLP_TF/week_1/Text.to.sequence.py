
# coding: utf-8

# # Text to sequence
# ## August 18th 2019

# In[4]:


import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# In[10]:


sentence= ['I love my dog', 'I love my cat', 'You love my dog!', 'Do you think my dog is amazing?']

tokenizer= Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentence)
#You need word index to make sense to train a NN
word_index= tokenizer.word_index

sequences= tokenizer.texts_to_sequences(sentence)


# In[11]:


print(word_index, "\n")
print(sequences)


# In[13]:


test_data= ['i really love my dog', 'my dog loves my manatee']

test_seq= tokenizer.texts_to_sequences(test_data)

# 'I really love my dog' it's encoded the same as 'I love my dog'
# The second one: my dog my 
print(test_seq)


# ## Using a different property of Tokenizer: 'OOV'

# In[25]:


sentence= ['I love my dog', 'I love my cat', 'You love my dog!', 'Do you think my dog is amazing?']

tokenizer= Tokenizer(num_words= 100, oov_token= "<OOV>") #OOV= outer vocabulary; to be used for words that aren't 
                                                         #in the word index
tokenizer.fit_on_texts(sentence)
word_index= tokenizer.word_index

print(word_index, "\n")

sequences= tokenizer.texts_to_sequences(sentence)
print(sequences, "\n")

test_data= ['i really love my dog', 'my dog loves my manatee']
test_seq= tokenizer.texts_to_sequences(test_data)
print("OOV: 1 ", test_seq)

