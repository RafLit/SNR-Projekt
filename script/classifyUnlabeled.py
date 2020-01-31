#!/usr/bin/env python3
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from createModel import create1Model, create2Model, create3Model

unlabeledPath = os.path.join('..','data','unlabeled')
acceptPath = os.path.join('..','data','result','accept')
rejectPath = os.path.join('..','data','result','reject')
if os.path.isdir(acceptPath):
    shutil.rmtree(acceptPath)
if os.path.isdir(rejectPath):
    shutil.rmtree(rejectPath)
os.makedirs(acceptPath)
os.makedirs(rejectPath)


if __name__ == '__main__':
    img_gen = ImageDataGenerator(rescale=1./255.)
    train_gen = img_gen.flow_from_directory('../data/train',target_size=(776,294), color_mode='rgb',
                                            class_mode='categorical', batch_size=5)
    img2_gen = ImageDataGenerator(rescale=1./255.)
    test_gen = img2_gen.flow_from_directory('../data/test', target_size=(776, 296), color_mode='rgb', batch_size=1,
                                            class_mode='categorical', shuffle=False)

    img3_gen = ImageDataGenerator(rescale=1./255.)
    unlabeled_gen = img3_gen.flow_from_directory(unlabeledPath, target_size=(776, 296),color_mode='rgb', batch_size=1,
                                                 class_mode=None, shuffle=False)

    model = create1Model()
    #model.summary()
    model.fit(train_gen, epochs=100)

    print('classifying data...')
    x = model.predict(unlabeled_gen)
    preds_classes = np.argmax(x, axis=-1)
    print('saving data...')
    for inPath, pred in zip(unlabeled_gen.filepaths, preds_classes):
        if pred:
            shutil.copy(inPath, os.path.join(rejectPath, os.path.split(inPath)[1]))
        else:
            shutil.copy(inPath, os.path.join(acceptPath, os.path.split(inPath)[1]))

