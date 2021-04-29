# Importing the libraries
from PIL import Image
import cv2
from keras.models import load_model

import numpy as np
import os
from datetime import datetime



# Loads the saved model
model = load_model(r'latestmodels/MaskNet27.2.h5')
# model = load_model(r'latestmodels/sfacial_recognition_dc5.h5')
# model = load_model(r'latestmodles/masknet5.h5')
# model = load_model(r'../latestmodles/modelvgg16v1.h5')
# model = load_model('latestmodels/sfacial_recognition_dc8v32.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')
# image = extract_face(file_path)
#   preprocessed_image = preprocess_input(image)
#   shaped_img = preprocessed_image.reshape(1,224,224,3)
#   classes = model1.predict(shaped_img)

names = ["Achintha", "Minindu", "Sandusha","Saneth", "Susith"]

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
    print('going to mark attendance')
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
        print(pred)

        name = "None matching"

        # threashold value is set.
        # if (pred[0][0] > 0.5):
        #     name = 'amma'
        #     cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # elif(pred[0][1] > 0.5):
        #     name = 'dad'
        #     cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        if (pred[0][4] > 0.10 ):
            val =pred[0][4]
            name = names[4]
            markAttendance(name)
            # y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            name = "no match found"
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # HERE WE HAVE TO GO WITH CLASS IDENTIFIER

    else:
        cv2.putText(frame, "face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
# while cv2.getWindowProperty('window-name', 0) >= 0:
#     keyCode = cv2.waitKey(50)
#     # ...



