import cv2 
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model

LABELS = ["a1","a2","a3","a4","a5","abo_elhgag","kapsh" , "masla" , "ship" 
          , "status1" , "status2" , "status3" , "status4","status5" ,
          "status6" , "status7" , "status8", "status9","wall1" , "wall2"]
def prepare(filepath):
  IMG_SIZE = 500
  img_array = cv2.imread(filepath)
  img_array = cv2.cvtColor(img_array ,cv2.COLOR_BGR2RGB)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def LoadModel(ModelPath):
    x = tf.keras.Input(shape=(500,500,3))
    y = tf.keras.layers.Dense(16, activation='softmax')(x)
    model = tf.keras.Model(x, y)
    model = load_model(ModelPath)
    return model 
#         'D:\gradProj\model.h5'
model = LoadModel('D:\gradProj\model.h5')

def predice_image(imagePath):
    prediction = model.predict([prepare('D:\gradProj\hima.jpg')])
    a = prediction
    i,j = np.unravel_index(a.argmax(), a.shape)
    a[i,j]
    return LABELS[j]
ans = predice_image('D:\gradProj\hima.jpg')
print('\n\n')
print(ans)
 
 
























