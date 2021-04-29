
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.applications.vgg16 import preprocess_input
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


import keras
# keras.__version__
# print(keras.__version__)


def extract_face(filename, required_size=(224, 224)):
  # load image from file
  image = Image.open(filename)
  # convert to RGB, if needed
  image = image.convert('RGB')
  # convert to array
  pixels = asarray(image)
  # create the detector, using default weights
  detector = MTCNN()
  # detect faces in the image
  results = detector.detect_faces(pixels)
  # extract the bounding box from the first face
  try:
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
  except IndexError:
    face_array=[]
    print("Out of index")

def load_faces(directory):
  faces = list()
  # enumerate files
  for filename in listdir(directory):
    # path
    path = directory + filename
    # get face
    face = extract_face(path)
    if face is not None:
      # store
      faces.append(face)
  return faces



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = directory + subdir + '/'
		# skip any files that might be in the dir
		if not isdir(path):
			continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)



# uncomment when using new dataset

# trainX, trainy = load_dataset('../dataset2/')
# print('datset loaded')
# print(trainX.shape, trainy.shape)
# # load test dataset
# # save arrays to one file in compressed format
# savez_compressed(r'..\util\checkpoints\sall_photos5_2.npz', trainX, trainy)
#
# print(trainX.shape)

npzpath = r'..\util/checkpoints/sall_photos5_2.npz';

data = np.load(npzpath)
x_train_cam, y_train_cam = data['arr_0'], data['arr_1']
print('Loaded: ', x_train_cam.shape, y_train_cam.shape)


print(np.unique(y_train_cam, return_counts=True))

# data augmentation

X, Y = data['arr_0'], data['arr_1']
print('Loaded: ', X.shape, Y.shape)

import cv2
import random
import numpy as np

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

import cv2
import random

def augment_data(imgs, scales, rotations, brightness_values, shifts, labels):
  timg=[]
  y_hat=[]
  for (img,label) in zip(imgs,labels):
    rows, cols, ch = img.shape
    for j in rotations:
      f = random.uniform(0, 1)
      #f = random.choice([0,1])
      rot_mat = cv2.getRotationMatrix2D( (14,14), j, 1);
      affined_img = cv2.warpAffine( img, rot_mat, (cols, rows), borderValue=(1,1,1))
      resulting_img = affined_img
      if f == 1:
        resulting_img = cv2.flip(resulting_img, f)
      resulting_img = brightness(resulting_img, brightness_values[0], brightness_values[1])
      timg.append(resulting_img)
      y_hat.append(label)
  return np.array(timg), np.array(y_hat)


scales = np.arange(0.9, 1, 0.1)
rotations = np.arange(0, 2)
brightness_values = [ 0.4, 2]
shifts = [ 0.2]
au_x, au_y = augment_data(X, scales, rotations, brightness_values, shifts, Y)


print(au_x.shape)

print(np.unique(au_y, return_counts=True))



savez_compressed(npzpath, x_train_cam, y_train_cam)
X, Y = data['arr_0'], data['arr_1']
print('Finally Loaded: ', X.shape, Y.shape)

# Preparing dataset: One-hot encoding

# split into train test sets
# au_x_train, au_x_test, au_y_train, au_y_val = train_test_split(x_train_cam, y_train_cam, test_size=0.33)
# print(au_x_train.shape, au_x_test.shape, au_y_train.shape, au_y_val.shape)

au_x_train, au_x_test, au_y_train, au_y_test = train_test_split(au_x, au_y, test_size=0.33)
print(au_x_train.shape, au_x_test.shape, au_y_train.shape, au_y_test.shape)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(au_y_train)
au_y_train = out_encoder.transform(au_y_train)
au_y_test = out_encoder.transform(au_y_test)

print(au_y_train.shape)

print(au_y_test.shape)


au_train_labels = to_categorical(au_y_train)
au_test_labels = to_categorical(au_y_test)

print(au_train_labels.shape)
print(au_test_labels.shape)

# The data one-hot encoded
savez_compressed(r'..\util\checkpoints\data_one_hot_encoded5_2.npz', au_x_train, au_train_labels, au_x_test, au_test_labels)

# load the faces dataset
data = np.load(r'..\util\checkpoints\data_one_hot_encoded5_2.npz')
au_x_train, au_train_labels, au_x_test, au_test_labels= data['arr_0'], data['arr_1'],data['arr_2'], data['arr_3']
print('Loaded: ', au_x_train.shape, au_train_labels.shape,  au_x_test.shape, au_test_labels.shape)

# Model from scratch

model1 = Sequential()

# Convolution layers
model1.add(Conv2D(32, (3,3), input_shape = (224, 224, 3), activation = 'relu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model1.add(MaxPooling2D(2))

model1.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(units = 64, activation = 'relu'))
model1.add(Dropout(0.2))

# Output
model1.add(Dense(units = 5, activation = 'softmax'))
model1.summary()

# Compiling the CNN
model1.compile(optimizer = 'adam',
                   loss='categorical_crossentropy',
                   metrics = ['accuracy'])

# model1.save('..\latestmodels\masknet_model_woes-shuff-f.h5')

early_stop = EarlyStopping(monitor='val_loss', patience = 3)
checkpoint = ModelCheckpoint('latestmodles/sfacial_recognition_dc5_aug_v1.h5', monitor = 'val_acc')

pre_au_x_train = preprocess_input(au_x_train)
pre_au_x_test = preprocess_input(au_x_test)

print(' final Loaded: ', pre_au_x_train.shape, au_train_labels.shape,  pre_au_x_test.shape, au_test_labels.shape)


#default batch size = 128

history = model1.fit(pre_au_x_train, au_train_labels,
                     batch_size = 32, epochs=20,
                     validation_data = (pre_au_x_test, au_test_labels),
                     callbacks = [early_stop, checkpoint])

# history = model1.fit(pre_au_x_train, au_train_labels,
#                      batch_size = 32, epochs=20,
#                      validation_data = (pre_au_x_test, au_test_labels)
#                      )

# from keras.models import load_model
# model1 = load_model(r'../latestmodels/sfacial_recognition_dc5.h5')

preds1 = model1.predict(pre_au_x_test, batch_size=50) # y_test are the truth, real labels correct ones.
# classes are the model predictions (That can be wrong or match the real ones)
print("Total predicted classes are:",len(preds1))



print(accuracy_score(au_test_labels.argmax(axis=1), preds1.argmax(axis=1)))


y_pred1 = np.argmax(preds1, axis=1)
print('Confusion Matrix')
print(confusion_matrix(au_test_labels.argmax(axis=1), y_pred1))
print('Classification Report')
target_names = ['s1', 's2', 's3','s4', 's5']
print(classification_report(au_test_labels.argmax(axis=1), y_pred1, target_names=target_names))




# # roc curve auc curve.
# import numpy as np
# from scipy import interp
# import matplotlib.pyplot as plt
# from itertools import cycle
# from sklearn.metrics import roc_curve, auc

# lw = 2
# n_classes=5;
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(au_y_val[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(au_y_val.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Compute macro-average ROC curve and ROC area
#
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure(1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()
#
#
# # Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()