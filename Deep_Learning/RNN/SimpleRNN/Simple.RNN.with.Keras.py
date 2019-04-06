
# coding: utf-8

# # Simple RNN with Keras
# ## April 5th 2019
# ### Build and train a RNN with Keras; information obtained from: Deep Learning with Keras 2017
# ### Page: 177 

# In[2]:


from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
# from keras.utils.visualize_util import plot
import numpy as np
import sys, os, re


# ## Preparing data input

# In[3]:


os.chdir('/Users/raziel/Documents/Deep-Learning-AI-Barcelona-2018/Deep_Learning/RNN/SimpleRNN')

fin= 'alice_wonderland.txt'
fin= open(fin, 'rb')

lines=[]
for line in fin:
    line= line.strip().lower()
    line= line.decode("ascii", "ignore")
    if len(line) == 0:
        continue
    lines.append(line)

fin.close()
text=" ".join(lines)
chars= set([i for i in text])
nb_chars= len(chars)

#i=counter, c=char value
char2index= dict((c,i) for i,c in enumerate(chars))
index2char= dict( (i,c) for i,c in enumerate(chars))

SEQLEN= 10
STEP= 1

input_chars= []
label_chars= []

for i in range(0, len(text)- SEQLEN, STEP):
    input_chars.append(text[i:i+ SEQLEN])
    label_chars.append(text[i + SEQLEN])


# ## Vectorize

# In[4]:


X= np.zeros((len(input_chars), SEQLEN, nb_chars), dtype= np.bool)
y= np.zeros((len(input_chars), nb_chars), dtype= np.bool)

for i, input_chars in enumerate(input_chars):
    for j, ch in enumerate(input_chars):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1


print(y.shape, "\n")
print(X.shape)


# ## Buld our model

# In[5]:


#Hyper-parameters: 
HIDDEN_SIZE= 128 #Experimental validation
BATCH_SIZE= 128
NUM_ITERARIONS= 25
NUM_EPOCHS_PER_ITERATION= 1
NUM_PREDS_PER_EPOCH= 100

#1) Architecture of the model:
model= Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences= False, input_shape= (SEQLEN, nb_chars), unroll= True))
model.add(Dense(nb_chars))
model.add(Activation('softmax'))

#2) Compile: Optimizer and loss function: 
model.compile(loss= "categorical_crossentropy", optimizer= "rmsprop")

#3) Fit the model:
#So far the training process was: to train a model for a fixed number of epochs then evaluate the model.
#Here we don't have labeled data 
for iterator in range(NUM_ITERARIONS):
    print('=' * 50) #"rep in R"
    print("Iterator #: %d" %(iterator))
    model.fit(X, y, batch_size= BATCH_SIZE, epochs= NUM_EPOCHS_PER_ITERATION)

    test_idx= np.random.randint(len(input_chars))
    test_chars= input_chars[test_idx]
    print("Generation from seed: %s" % (test_chars))
    print(test_chars, end="")

    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest= np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        #4) Model Predict: 
        pred= model.predict(Xtest, verbose= 0)[0]
        ypred= index2char[np.argmax(pred)]
        print(ypred, end="")
        # move forward with test_chars + ypred
        test_chars= test_chars[1:] + ypred
print()

