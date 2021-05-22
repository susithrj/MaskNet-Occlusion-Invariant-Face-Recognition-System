# Importing the libraries
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
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import random


'''
Functions to extract faces using MTCNN.
'''
def extract_face(filename, required_size=(224, 224)):

  image = Image.open(filename)
  image = image.convert('RGB')
  pixels = asarray(image)
  detector = MTCNN()
  results = detector.detect_faces(pixels)

  try:
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
  except IndexError:
    face_array=[]
    print("Out of index")
    print(face_array.index())

def load_faces(directory):
  faces = list()
  for filename in listdir(directory):
    path = directory + filename
    face = extract_face(path)
    if face is not None:
      faces.append(face)
  return faces

def load_dataset(directory):
	X, y = list(), list()
	for subdir in listdir(directory):
		path = directory + subdir + '/'

		if not isdir(path):
			continue

		faces = load_faces(path)
		labels = [subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)


# uncomment when using new dataset
trainX,trainy = load_dataset('dataset5/train/')
valX,valY  = load_dataset('dataset5/val/')
testX,testY  = load_dataset('dataset5/test/')
print('dataset loaded')
print(trainX.shape, trainy.shape)

savez_compressed(r'util\checkpoints\new\train_sall_photos300_1.npz', trainX, trainy)
savez_compressed(r'util\checkpoints\new\val_sall_photos300_1.npz', valX, valY)
savez_compressed(r'util\checkpoints\new\test_sall_photos300_1.npz', testX, testY)
print(trainX.shape)
print(testX.shape)
print(valX.shape)

npzpathtrain = r'util\checkpoints\new\train_sall_photos300_1.npz';
npzpathtest = r'util\checkpoints\new\val_sall_photos300_1.npz';
npzpathval = r'util\checkpoints\new\test_sall_photos300_1.npz';

data = np.load(npzpathtrain)
x_train_cam, y_train_cam = data['arr_0'], data['arr_1']
print('Loaded: ', x_train_cam.shape, y_train_cam.shape)

print(np.unique(y_train_cam, return_counts=True))

data = np.load(npzpathval)
x_val_cam, y_val_cam = data['arr_0'], data['arr_1']
print('Loaded: ', x_val_cam.shape, y_val_cam.shape)

print(np.unique(y_val_cam, return_counts=True))

data = np.load(npzpathtest)
x_test_cam, y_test_cam = data['arr_0'], data['arr_1']
print('Loaded: ', x_test_cam.shape, y_test_cam.shape)

print(np.unique(y_test_cam, return_counts=True))

print('before aug',x_train_cam.shape, x_val_cam.shape, x_test_cam.shape, y_train_cam.shape, y_val_cam.shape, y_val_cam.shape)

'''
Functions to augment data starts.
'''
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

def augment_data(imgs, rotations, brightness_values, labels):
  timg=[]
  y_hat=[]
  for (img,label) in zip(imgs,labels):
    rows, cols, ch = img.shape
    for j in rotations:
      f = random.uniform(0, 1)
      rot_mat = cv2.getRotationMatrix2D( (14,14), j, 1);
      affined_img = cv2.warpAffine( img, rot_mat, (cols, rows), borderValue=(1,1,1))
      resulting_img = affined_img
      if f == 1:
        resulting_img = cv2.flip(resulting_img, f)
      resulting_img = brightness(resulting_img, brightness_values[0], brightness_values[1])
      timg.append(resulting_img)
      y_hat.append(label)
  return np.array(timg), np.array(y_hat)



rotations = np.arange(0, 2)
brightness_values = [ 0.4, 2]


au_train_x, au_train_y = augment_data(x_train_cam,  rotations, brightness_values, y_train_cam)
au_val_x, au_val_y = augment_data(x_val_cam, rotations, brightness_values, y_val_cam)
au_test_x, au_test_y = augment_data(x_test_cam,rotations, brightness_values, y_test_cam)

print(au_train_x.shape)
print(np.unique(au_train_y, return_counts=True))

print(au_val_x.shape)
print(np.unique(au_val_y, return_counts=True))

print(au_test_x.shape)
print(np.unique(au_test_y, return_counts=True))
print( 'FINAL AUG',au_train_x.shape, au_val_x.shape, au_test_x.shape, au_train_y.shape, au_val_y.shape,au_test_y.shape)

'''
Functions to one-hot encoding.
'''

out_encoder = LabelEncoder()
out_encoder.fit(au_train_y)
out_encoder.fit(au_val_y)
au_y_train = out_encoder.transform(au_train_y)
au_y_val = out_encoder.transform( au_val_y)
au_y_test = out_encoder.transform( au_test_y)

print(au_y_train.shape)
print(au_y_val.shape)
print(au_y_test.shape)

au_train_labels = to_categorical(au_y_train)
au_val_labels = to_categorical(au_y_val)
au_test_labels = to_categorical(au_y_test)

print(au_train_labels.shape)
print(au_val_labels.shape)
print(au_test_labels.shape)

savez_compressed(r'util/checkpoints/new/data_one_hot_encoded5_train&val&test_aug_1.npz', au_train_x, au_train_labels,au_val_x, au_val_labels,au_test_x,au_test_labels)
data = np.load(r'util/checkpoints/new/data_one_hot_encoded5_train&val&test_aug.npz')
au_x_train, au_train_labels, au_x_val, au_val_labels, au_x_test, au_test_labels = data['arr_0'], data['arr_1'],data['arr_2'], data['arr_3'],data['arr_4'], data['arr_5']
print('Loaded encoded FINAL AUG: ', au_x_train.shape, au_train_labels.shape,  au_x_val.shape, au_val_labels.shape, au_x_test.shape, au_test_labels.shape)


'''
Comprehensive Feature Learning model
'''

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

#Compiling
model1.compile(optimizer = 'adam',
                   loss='categorical_crossentropy',
                   metrics = ['accuracy'])

#Early checkpoints
early_stop = EarlyStopping(monitor='val_loss', patience = 3)
checkpoint = ModelCheckpoint('latestmodels\MaskNet.h5', monitor = 'val_acc')

pre_au_x_train = preprocess_input(au_x_train)
pre_au_x_val = preprocess_input(au_x_val)
pre_au_x_test = preprocess_input(au_x_test)

print(' finally Loaded: ', pre_au_x_train.shape, au_train_labels.shape,  pre_au_x_val.shape, au_val_labels.shape)


history = model1.fit(pre_au_x_train, au_train_labels,
                     batch_size = 32, epochs=20,
                     validation_data = (pre_au_x_val, au_val_labels),
                     callbacks = [early_stop, checkpoint])

'''
results are starts here.
'''

# plot model acc val acc
# import matplotlib.pyplot as plt
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# plot model loss and val loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
#
# preds1 = model1.predict(pre_au_x_test, batch_size=50)
# print("Total predicted classes are:",len(preds1))
#
# print(accuracy_score(au_val_labels.argmax(axis=1), preds1.argmax(axis=1)))
# y_pred1 = np.argmax(preds1, axis=1)
#
# print('Confusion Matrix')
# print(confusion_matrix(au_val_labels.argmax(axis=1), y_pred1))
#
# print('Classification Report')
# target_names = ['s1', 's2', 's3','s4', 's5']
# print(classification_report(au_val_labels.argmax(axis=1), y_pred1, target_names=target_names))



