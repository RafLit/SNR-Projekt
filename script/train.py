#!/usr/bin/env python3
import numpy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib
#from create import createModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten


if __name__ == '__main__':
  print('Hello world!')
  path = pathlib.Path('../data/accept')
  image_count = len(list(path.glob('*.bmp')))
  roses = list(path.glob('*.bmp'))
  im = Image.open(str(roses[0]))
  print(im)
  train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True)
  train_generator = train_image_generator.flow_from_directory('../data/train',
                                            target_size=(776,294),
                                            color_mode='rgb',
                                            batch_size=8,
                                            class_mode='categorical',
                                          )

  model = Sequential()
  model.add(Conv2D(32, (9,5), input_shape=(776, 294, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (5,5),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, (3,3), activation='relu'))


  model.add(Flatten())
  model.add(Dense(32, activation='relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(2, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_generator, epochs=30)
  #model.fit(train_generator, epochs=30)
