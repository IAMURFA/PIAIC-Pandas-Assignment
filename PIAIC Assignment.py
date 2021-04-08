#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.datasets import fashion_mnist


# In[ ]:


(train_images,train_labels) , (test_images, test_lables) = fashion_mnist.load_data()


# In[ ]:


print(len(train_images))
print(len(train_labels))


# In[ ]:


print(len(test_images))
print(len(test_lables))


# In[ ]:


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_lables.shape)


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[ ]:


import matplotlib.pyplot as plt
print("class name" , class_names[train_labels[15]])
plt.imshow(train_images[15] , cmap=plt.cm.binary)
plt.show()


# In[ ]:


import numpy as np
print(np.unique(train_labels))
print(np.unique(test_lables))


# In[ ]:


print(train_images.shape)
train_images=train_images.reshape((60000, 28*28))
print(train_images.shape)
train_images=train_images.astype("float32")/255
print(train_images.shape)

test_images=test_images.reshape((10000, 28*28))
test_images=test_images.astype("float32")/255


# In[ ]:


train_labels[2]


# In[ ]:


from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_lables = to_categorical(test_lables)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten , Dropout
from tensorflow.keras.activations import relu ,softmax
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.losses import categorical_crossentropy


# In[ ]:


network = Sequential()
network.add(Dense(512 , activation=relu , input_shape=(28*28,)))
network.add(Dropout(0.2))
network.add(Dense(10 , activation=softmax))


# In[ ]:


network.summary()


# In[ ]:


network.compile(optimizer='rmsprop' , loss='categorical_crossentropy' , metrics=['acc'])


# In[ ]:


train_images.shape


# In[ ]:


history=network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[ ]:


print(history.history.keys())


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy and loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc','trian loss'], loc='upper left')
plt.show()


# In[ ]:


test_loss, test_acc = network.evaluate(test_images , test_lables)


# In[ ]:


print("losss" , test_loss)


# In[ ]:


print("model acc" , test_acc)

