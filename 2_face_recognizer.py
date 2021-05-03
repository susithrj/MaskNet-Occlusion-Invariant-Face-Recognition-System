# Importing the libraries
from PIL import Image
import cv2
from keras.models import load_model

import numpy as np
import os
from datetime import datetime

import os

import os
import os




# Loads the saved daytime model
# model = load_model(r'latestmodels/MaskNet27.2.h5')

# Loads the saved nt model
#model = load_model(r'latestmodels/MaskNet27.1.h5')

# Loads the saved b model
model = load_model(r'latestmodels/MaskNet.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')
# image = extract_face(file_path)
#   preprocessed_image = preprocess_input(image)
#   shaped_img = preprocessed_image.reshape(1,224,224,3)
#   classes = model1.predict(shaped_img)

# names = ["Achintha", "Minindu", "Sandusha","Saneth", "Susith"]
names = os.listdir('dataset5/train')
# print(names)
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)


def markAttendance(name):
    # print('going to mark attendance')
    with open('attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        # print(pred)



        # threashold value is set.
        # if (pred[0][0] > 0.5):
        #     name = 'achintha'
        #     cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # elif(pred[0][1] > 0.5):
        #     name = 'dad'
        #     cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # threashold value is set.
        if (pred[0][4] > 0.90 ):
            val =pred[0][4]
            name = names[4]
            markAttendance(name)
            att = 'Your attendance marked'
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, att, (40, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        else:
            name = "no match found"
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # HERE WE HAVE TO GO WITH CLASS IDENTIFIER

    else:
        cv2.putText(frame, "face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27 :  # 13 is the Enter Key 27 esc
        break
video_capture.release()
cv2.destroyAllWindows()



