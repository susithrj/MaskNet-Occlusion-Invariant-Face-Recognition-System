# Importing the libraries
from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
from datetime import datetime
import os

# Loads model
model = load_model(r'latestmodels/MaskNet.h5')

# Loads haarcascade
face_cascade = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')

#Loding the names of datadir
names = os.listdir('dataset5/train')

'''
Function to extract faces.
'''

def face_extractor(img):

    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face

'''
Function to mark attendance.
'''
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

'''
Real time recognition starts here.
'''
# Initialize camera

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        #threashold value
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


    else:
        cv2.putText(frame, "face not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27 :  # 27 = Esc Key
        break
video_capture.release()
cv2.destroyAllWindows()



