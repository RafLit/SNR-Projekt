#!/usr/bin/env python3
import numpy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
import pathlib
from sklearn.metrics import roc_curve, confusion_matrix, plot_roc_curve, auc
import numpy as np
#from create import createModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten
from sklearn.metrics import roc_auc_score
import os
import shutil

unlabeledPath = os.path.join('..','data','unlabeled','unlabeled')
acceptPath = os.path.join('..','data','unlabeled','accepted')
rejectPath = os.path.join('..','data','unlabeled','reject')
if os.path.isdir(acceptPath):
    shutil.rmtree(acceptPath)
if os.path.isdir(rejectPath):
    shutil.rmtree(rejectPath)
os.makedirs(acceptPath)
os.makedirs(rejectPath)
print('deleted')


if __name__ == '__main__':
    img_gen = ImageDataGenerator(rescale=1./255.)
    train_gen = img_gen.flow_from_directory('../data/train',target_size=(776,294), color_mode='rgb', class_mode='categorical'
                                            , batch_size=5)
    img2_gen = ImageDataGenerator(rescale=1./255.)
    test_gen = img2_gen.flow_from_directory('../data/test', target_size=(776, 296), color_mode='rgb', batch_size=1, class_mode='categorical', shuffle=False)

    img3_gen = ImageDataGenerator(rescale=1./255.)
    unlabeled_gen = img3_gen.flow_from_directory('../data/unlabeled/unlabeled', target_size=(776, 296),color_mode='rgb', batch_size=1, class_mode=None, shuffle=False)


    model = Sequential()
    model.add(Conv2D(64, (7,7), input_shape=(776, 294, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5)))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(500, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_gen, epochs=30)

    x = model.predict(unlabeled_gen)
    preds_classes = np.argmax(x, axis=-1)
    print(preds_classes)
    print(sum(preds_classes))
    for inPath, pred in zip(unlabeled_gen.filepaths, preds_classes):
        if pred:
            shutil.copy(inPath, os.path.join(rejectPath, os.path.split(inPath)[1]))
        else:
            shutil.copy(inPath, os.path.join(acceptPath, os.path.split(inPath)[1]))

