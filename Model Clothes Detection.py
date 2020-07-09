#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras


# In[17]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[18]:


train_images.shape


# In[19]:


len(train_labels)


# In[20]:


class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# In[21]:


class_names


# In[42]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)


# In[23]:


train_images = train_images/255.0
test_images = test_images/255.0


# In[27]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks()
    plt.yticks()
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[32]:


model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])


# In[33]:


model.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy', 
             metrics = ['accuracy'])


# In[34]:


model.fit(train_images,train_labels, epochs=5)


# In[36]:


prediction = model.predict(test_images)


# In[43]:


prediction[0]


# In[ ]:




