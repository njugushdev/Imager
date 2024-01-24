#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mp1
import matplotlib.pyplot as plt


# # Building an image classifier

# First lets install and import Tensorflow and Keras

# conda install tensorflow
# 
# conda install pip
# 
# pip install --upgrade tensorflow==2.0.0-rc1

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


keras.__version__


# In[4]:


tf.__version__


# # Usage

# from keras.datasets import fashion mnist
# 
#         (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
# Returns 2 Tuples:
# 
#    1.  x_train, x_test - uint8 array of greyscale image data with numsamples, 28,28
#    2.  y_train, y_test - uint8 array of labels(integers from range 1-9) with shape(numsamples)

# In[5]:


fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full),(x_test, y_test) = fashion_mnist.load_data()


# In[13]:


plt.imshow(x_train_full[0])


# In[14]:


y_train_full[0]


# In[10]:


class_names = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# In[15]:


class_names[y_train_full[10]]


# In[12]:


x_train_full[1]


# # Data normalization

# We normalize the data dimensions so that they are approximately the same scale

# In[16]:


x_train_n = x_train_full / 255.
x_test_n = x_test / 255.


# # Split the Data into Train/Validation/Test Datasets

# In the earlier step of importing we has 60,000 train datasets and 10,000 test datasets. Now we further split the training data into Train/Validation. Here's how each of the datasets is used in deep learning:
# 
#     -Training Data -- used for training the model
#     -Validation Data -- used for tuning the hyperparameters and evaluate the models
#     -Test Data -- Used to test the model after it has gone through initial vetting by validation set
# 

# In[17]:


x_valid, x_train = x_train_n[:5000], x_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test_n


# # Building a sequential model

# In[18]:


np.random.seed(42)
tf.random.set_seed(42)


# In[19]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))


# In[20]:


model.summary()


# In[21]:


import pydot
keras.utils.plot_model(model)


# In[22]:


weights, biases = model.layers[1].get_weights()


# In[23]:


weights


# In[24]:


weights.shape


# In[25]:


biases


# In[27]:


biases.shape


# # Compiling and training

#  Before training model, we need to compile the data.
#  
#  Documentation: https://keras.io/api/models/sequential/

# In[28]:


model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = "sgd",
             metrics = ["accuracy"])


# In[29]:


model_history = model.fit(x_train, y_train, epochs = 30,
                         validation_data = (x_valid, y_valid))


# In[30]:


model_history.params


# In[31]:


model_history.history


# In[32]:


import pandas as pd
pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid = True
plt.gca().set_ylim(0, 1)
plt.show()


#  # Evaluating performance and predicting

# In[33]:


model.evaluate(x_test, y_test)


# # Predicting 

# In[34]:


x_new = x_test[:3]


# In[35]:


y_proba = model.predict(x_new)
y_proba.round(2)


# In[36]:


y_pred = model.predict_classes(x_new)
y_pred


# In[37]:


np.array(class_names)[y_pred]


# In[38]:


print(plt.imshow(x_test[0]))


# In[39]:


print(plt.imshow(x_test[1]))


# In[40]:


print(plt.imshow(x_test[2]))


# In[ ]:





# In[ ]:





# In[ ]:




