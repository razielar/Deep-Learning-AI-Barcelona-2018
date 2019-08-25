#!/usr/bin/env python
# coding: utf-8

# # Kaggle notebook for Sarcasm detection
# ## August 25th 2019 
# ## [Kaggle Notebook link](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection#Sarcasm_Headlines_Dataset.json)

# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[23]:


df = pd.read_json("sarcasm.json", lines= True)
df.head()


# In[24]:


sns.countplot(df.is_sarcastic)
plt.xlabel('Label')
plt.title('Sarcasm Detection')


# In[37]:


sentences= list(df.headline)
### Generate the Tokenizer:
tokenizer= Tokenizer(oov_token= '<OOV>')
tokenizer.fit_on_texts(sentences)
word_index= tokenizer.word_index

sequences= tokenizer.texts_to_sequences(sentences)
padded= pad_sequences(sequences=sequences, padding= 'post') #put '0' after the sentence
print("Word index length: ",len(word_index), "\n")
print("Sentence: \n",sentences[2], "\n")
print("Padded: \n",padded[2], "\n")
print(padded.shape)


# ### Process to convert text to data: 
# 1. Create **Tokenizer** object
# 2. Fit on text
# 3. Create **Word index**
# 4. Text to sequence
# 5. Padd

# In[ ]:




