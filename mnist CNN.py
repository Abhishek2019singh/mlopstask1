#!/usr/bin/env python
# coding: utf-8

# In[49]:


from keras.datasets import mnist


# In[50]:


(X_train,y_train),(X_test,y_test) = mnist.load_data()


# In[51]:


X_train=X_train.reshape(60000,28,28,1)


# In[52]:


X_test=X_test.reshape(10000,28,28,1)


# In[53]:


from keras.utils import to_categorical


# In[55]:


y_train = to_categorical(y_train)


# In[56]:


y_test = to_categorical(y_test)


# In[32]:


from keras.models import Sequential


# In[57]:


from keras.layers import Dense, Conv2D,Flatten


# In[58]:


model = Sequential()


# In[62]:


model.add(Conv2D(2, kernel_size=3, activation="relu",input_shape=(28,28,1)))


# In[63]:


i=1
n=4


# In[67]:


for i in range(i):
    model.add(Conv2D(filters=n,kernel_size=3,activation="relu"))
    n=n*2
    


# In[68]:


model.add(Flatten())


# In[69]:


model.add(Dense(10,activation="softmax"))


# In[72]:


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[73]:


model.fit(X_train,y_train,epochs=1)


# In[74]:


pred=model.evaluate(X_test,y_test)


# In[75]:


print("Accuracy is :-",pred[1]*100)


# In[ ]:




