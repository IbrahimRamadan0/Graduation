from keras.models import load_model


# returns a compiled model
# identical to the previous one
model = load_model('/content/drive/MyDrive/semifinalmodel/model.h5')

img = "/content/drive/MyDrive/dataset/a1/a1_1.jpg

model.predict();

import cv2 
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model
LABELS = ["a1","a2","a3","a4","a5","abo_elhgag","kapsh" , "masla" , "ship" , "status1" , "status2" , "status3" , "status4","status5" , "status6" , "status7" , "status8", "status9","wall1" , "wall2"]
def prepare(filepath):
  IMG_SIZE = 500
  img_array = cv2.imread(filepath)
  img_array = cv2.cvtColor(img_array ,cv2.COLOR_BGR2RGB)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x = tf.keras.Input(shape=(500,500,3))
y = tf.keras.layers.Dense(16, activation='softmax')(x)
model = tf.keras.Model(x, y)
model = load_model('/content/drive/MyDrive/semifinalmodel/نسخة من model.h5')

prediction = model.predict([prepare('/content/drive/MyDrive/luxor/478306_0.jpg')])
a = prediction
i,j = np.unravel_index(a.argmax(), a.shape)
a[i,j]
print(LABELS[j])

