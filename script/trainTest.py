#!/usr/bin/env python3
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from createModel import create1Model, create2Model, create3Model

ep = [1,10,100]
auce = []
confs = []
train_confs = []

for epch in ep:
  img_gen = ImageDataGenerator(rescale=1./255., vertical_flip=True)
  train_gen = img_gen.flow_from_directory('../data/train',target_size=(776,294), color_mode='rgb',
                                          class_mode='categorical', batch_size=3)

  img2_gen = ImageDataGenerator(rescale=1./255.)
  test_gen = img2_gen.flow_from_directory('../data/test', target_size=(776, 294), color_mode='rgb', batch_size=1,
                                          class_mode='categorical', shuffle=False)

  model = create1Model()
  # model.summary()
  model.fit(train_gen, epochs=epch)
  n_classes = 2
  y_test = test_gen.classes
  score = model.evaluate(test_gen)
  y_score = model.predict(test_gen)
  a,b,_= roc_curve(test_gen.classes, y_score[:,1])
  auce.append(roc_auc_score(test_gen.classes, y_score[:,1]))
  plt.plot(a,b)
  preds_classes = np.argmax(y_score, axis=-1)
  confs.append(confusion_matrix(test_gen.classes, preds_classes))

plt.plot([0,1],[0,1])
plt.title('Wykres ROC')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(['1 epoka','10 epok','100 epok'])
plt.show()
#plt.savefig('roc_curve.png')
print(auce)
print(confs)
print(train_confs)

