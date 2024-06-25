#!/usr/bin/env python
# coding: utf-8

# In[74]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np
import pandas as pd


# In[75]:


data=pd.read_csv('dataset_01.csv')


# In[76]:


data_1=data[data['Usage']=='Training']


# In[77]:


data_2=data[data['Usage']!='Training']


# In[78]:


data_1.to_csv('train_data_cnn.csv', index=False)


# In[79]:


data_2.to_csv('test_data_cnn.csv', index=False)


# In[ ]:





# In[80]:


# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator()



# In[81]:


from tensorflow.keras.utils import to_categorical
data1 = pd.read_csv('train_data_cnn.csv')
data2 = pd.read_csv('test_data_cnn.csv')

def preprocess_data(data):
    X = np.array([np.fromstring(pixels, dtype=int, sep=' ').reshape(48, 48, 1) for pixels in data['pixels']])
    y = to_categorical(data['emotion'], num_classes=7)
    return X, y

X_train, y_train = preprocess_data(data1)
X_test, y_test = preprocess_data(data2)

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[ ]:





# In[ ]:





# In[82]:


train_generator = train_datagen.flow(X_train,y_train, batch_size=64)
test_generator = test_datagen.flow(X_test,y_test, batch_size=64)


# In[83]:


emotion_model=Sequential()
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
emotion_model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))


# In[84]:


emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))


# In[85]:


emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7,activation='softmax'))


# In[86]:


emotion_model.summary()


# In[87]:


initial_learning_rate=0.0001
lr_schedule=ExponentialDecay(initial_learning_rate,decay_steps=100000,decay_rate=0.96)
optimizer=Adam(learning_rate=lr_schedule)
emotion_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


# In[88]:


emotion_model_info=emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709//128,
    epochs=50,
    validation_data=test_generator,
    validation_steps=7178//128)


# In[89]:


emotion_model.evaluate(test_generator)


# In[90]:


accuracy=emotion_model_info.history['accuracy']
val_accuracy=emotion_model_info.history['val_accuracy']
loss=emotion_model_info.history['loss']
val_loss=emotion_model_info.history['val_loss']


# In[91]:


print(val_accuracy)


# In[ ]:





# In[92]:


import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(accuracy,label='accuracy')
plt.plot(val_accuracy,label='val_accuracy')
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[93]:


plt.subplot(1,2,2)
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




