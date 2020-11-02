#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


label_dict={"with_mask":0, "without_mask":1}  #dictionary


# In[3]:


categories=["with_mask","without_mask"]       #list


# In[4]:


label=[0,1]


# In[5]:


data_path="C:\\Users\\anush\\Documents\\dataset"         


# In[6]:


import cv2,os


# In[7]:


data=[]    
target=[]     #empty lists


# In[8]:


for category in categories:
  folder_path=os.path.join(data_path,category)
  img_names=os.listdir(folder_path)
  for img_name in img_names:
    img_path=os.path.join(folder_path,img_name)
    img=cv2.imread(img_path)
    try:
      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      resized=cv2.resize(gray,(100,100))
      data.append(resized)
      target.append(label_dict[category])
    except Exception as e:
      pass


# In[9]:


import numpy as np
data=np.array(data)
data=data/255.0


# In[10]:


data


# In[11]:


data.shape


# In[12]:


data=np.reshape(data,(data.shape[0],100,100,1))


# In[13]:


data.shape


# In[14]:


target=np.array(target)


# In[15]:


target.shape


# In[16]:


from keras.utils import np_utils


# In[17]:


new_target=np_utils.to_categorical(target)


# In[18]:


new_target.shape


# In[19]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D,MaxPooling2D


# In[20]:


model = Sequential()
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:], activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))


# In[21]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


train_data,test_data,train_target,test_target =train_test_split(data,new_target,test_size=0.1)


# In[24]:


from keras.callbacks import ModelCheckpoint


# In[25]:


checkpoint=ModelCheckpoint("model-{epoch:03d}.model", save_best_only=True,mode="auto")
history=model.fit(train_data,train_target,epochs=30,validation_split=0.2,callbacks=[checkpoint])


# In[26]:


face_cascader=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# In[27]:


img=cv2.imread("C:\\Users\\anush\\Desktop\\Anushka.jpeg")


# In[28]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascader.detectMultiScale(img,1.3,5)  


# In[29]:


faces


# In[30]:


labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


# In[37]:


source=cv2.VideoCapture(0)
while(True):

    ret,img=source.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascader.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        #normalized=resized/255.0
        
        #result=model.predict(normalized)
        normimage=resized/255
        reshapeimage=np.reshape(normimage,(-1,100,100,1))
        modelop=model.predict(reshapeimage)
        
        label=np.argmax(modelop,axis=1)[1]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],1)
        
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
       # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
       # cv2.rectangle(img,(x,y-40),(x+w,y),(0,0,255),1)
        
        #cv2.putText(img, "face", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow("checking...",img)
    key=cv2.waitKey(2)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()

