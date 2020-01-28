#!/usr/bin/env python3
import numpy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib
from sklearn.metrics import roc_curve, confusion_matrix, plot_roc_curve, auc, roc_auc_score
import numpy as np
#from create import createModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from sklearn.metrics import roc_auc_score

ep = [1,10, 100]
auce = []
confs = []
train_confs = []

for epch in ep:
  img_gen = ImageDataGenerator(rescale=1./255.)
  train_gen = img_gen.flow_from_directory('../data/train',target_size=(776,294), color_mode='rgb', class_mode='categorical'
                                          , batch_size=3
                                          #,shuffle=False
                                          )
  img2_gen = ImageDataGenerator(rescale=1./255.)
  test_gen = img2_gen.flow_from_directory('../data/test', target_size=(776, 294), color_mode='rgb', batch_size=1, class_mode='categorical', shuffle=False)




  model = Sequential()
  model.add(Conv2D(64, (5,5), input_shape=(776, 294, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))
  model.add(Conv2D(128, (5,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))
  model.add(Conv2D(64, (5,5), activation='relu'))
  model.add(Conv2D(64, (5,3), activation='relu'))
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))
  model.add(Conv2D(64, (5,3), activation='relu'))
  model.add(Conv2D(128, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3)))

  model.add(Flatten())
  model.add(Dense(2000, activation='relu'))
  model.add(Dense(500, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  model.fit(train_gen, epochs=epch)
  print('evaluate')
  n_classes = 2
  y_test = test_gen.classes
  score = model.evaluate(test_gen)
  y_score = model.predict(test_gen)
  a,b,_= roc_curve(test_gen.classes, y_score[:,1])
  auce.append(roc_auc_score(test_gen.classes, y_score[:,1]))


  plt.plot(a,b)

  preds_classes = np.argmax(y_score, axis=-1)
  confs.append(confusion_matrix(test_gen.classes, preds_classes))

  #yt_score = model.predict(train_gen)
  #predst_classes = np.argmax(yt_score, axis=-1)
  #train_confs.append(confusion_matrix(train_gen.classes, predst_classes))

plt.plot([0,1],[0,1])
plt.title('Wykres ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(['1 epoka','10 epok','100 epok'])
plt.savefig('roc_curve.png')
print(auce)
print(confs)
print(train_confs)

