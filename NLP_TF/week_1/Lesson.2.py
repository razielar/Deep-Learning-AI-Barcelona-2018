
# coding: utf-8

# # Lesson 2
# ## August 18th 2019

# In[4]:


#Import modules:
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(tf.__version__)


# In[20]:


sentences= ['I love my dog', 'I love my cat', 'You love my dog!', 'Do you think my dog is amazing?']

tokenizer= Tokenizer(num_words=100, oov_token= "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index= tokenizer.word_index

sequences= tokenizer.texts_to_sequences(sentences)
padd= pad_sequences(sequences= sequences, maxlen= 8)

print("Word index: \n", word_index, "\n")
print("Sequences: \n", sequences, "\n")
print("Padded: \n", padd, "\n")

### Try with words that the tokenizer wan't fit to
test_data= ['i really love my dog', 'my dog loves my manatee']
test_seq= tokenizer.texts_to_sequences(test_data)
padded= pad_sequences(test_seq, maxlen= 10)

print("Test sequence: \n", test_seq, "\n")
print("Padded test sequence: \n",padded)

