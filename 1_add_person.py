# import statements
import cv2
import os
import sqlite3
from tkinter import *

'''
Function to assure whether the folder to store the training images are here.

'''
def assure_folder_exists(folder):
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Variables
news_sticky = N + E + W + S
bg_color = "#ADD8E6"
fg_color = "black"
config_color = '#A4CCD0'
col_num = 1

# UI Elements
main_frame = Tk()
main_frame.title("Registration")
main_frame.configure(bg=config_color)

# Labels
var_1 = StringVar()
l_1 = Label(main_frame, textvariable=var_1, bg=bg_color, fg=fg_color, relief=RAISED)
var_1.set("Nametest:")
l_1.grid(row=2, column=col_num, sticky=news_sticky)

# Inputs
e1_val = StringVar()
e_1 = Entry(main_frame, textvariable=e1_val, bg=bg_color, fg=fg_color)
e_1.grid(row=2, column=col_num + 1)

# UI functions
def pass_inputs():
    return e1_val.get()

def complete_information_gathering():
    main_frame.destroy()

# Buttons
b1 = Button(main_frame, text="OK", bg=bg_color, fg=fg_color, command=complete_information_gathering)
b1.grid(row=10, column=col_num+1)

main_frame.mainloop()
# End of UI functions

'''
The rest of this code is procedural programming.
'''

# face capturing start
# Load HAAR cascade face detector
face_classifier = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face

def assure_folder_exists(folder):
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

#intialize directory
# get person name into program
nametest = pass_inputs()
print("created directory for name: "+nametest)
#making directory for new person
os.makedirs('dataset4/train/'+nametest)
assure_folder_exists('dataset4/train/'+ nametest)
# os.makedirs('dataset4/train/'+ nametest)
# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0
# Collect 100 samples of your face from camera input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'dataset4/train/'+nametest+'/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")












# parameters = (int(id), name, age, email, address)
# #######
#
# sql_create = '''CREATE TABLE IF NOT EXISTS Persons
#          (ID INT PRIMARY KEY     NOT NULL,
#          NAME           TEXT    NOT NULL,
#          DOB            TEXT     NOT NULL,
#          EMAIL        CHAR(50),
#          ADDRESS    TEXT    CHAR(50));'''
# sql_select = "SELECT * from Persons"
#
# connection = sqlite3.connect('database/persons.db')
# connection.cursor()
# connection.execute(sql_create)
#
# sql = "INSERT INTO Persons (ID,NAME,DOB,EMAIL,ADDRESS) VALUES (?, ?, ?, ?,?)"
# connection.execute(sql, parameters)
# connection.execute(sql_select)
# connection.commit()
# connection.close()
#
#
# # Concurrent variables
# face_id = int(id)
# name_id = name
#
# # Video - webcam start
# web_cam = cv2.VideoCapture(0)
#
# # Detect front face using HAARCASCADE FF
# init_face_crop = cv2.CascadeClassifier('dir_util/haarcascade_frontalface_default.xml')
#
# # Face Count
# num_of_faces = 0
#
# assure_folder_exists("images/")
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# # encryption keys
# enc = 'Spubuf!zpvs!gbdf!Dmpdlxjtf'
#
# def decrypt(kj):
#     fr = []
#     for i in kj:
#         fr.append(chr(ord(i)-1))
#     return "".join(fr)
#
# # Loop for faces until num of faces saved.
# while (True):
#
#     # Analyse Web cam video feed
#     _, single_image = web_cam.read()
#
#     # Remove color channels
#     remove_color_channel = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)
#
#     # Detect number of faces in the image
#     list_of_faces = init_face_crop.detectMultiScale(remove_color_channel, 1.3, 5)
#
#     # for each face in list_of_faces
#     for (x, y, w, h) in list_of_faces:
#         # Crop and vectorise the image
#         cv2.rectangle(single_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         # label text
#         cv2.putText(single_image, decrypt(enc), (x, y - 10), font, 0.5, (120, 255, 120), 2, 1)
#
#         # num of faces
#         num_of_faces += 1
#
#         if num_of_faces == 1:
#             # create training data.
#             cv2.imwrite("images/" + name_id +"_" + str(face_id) +".jpg", remove_color_channel[y:y + h, x:x + w])
#
#         # Display face with bouding boxes
#         cv2.imshow('frame', single_image)
#
#     # Stop video frame press q
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     elif num_of_faces >= 30:
#         print("Successfully Captured")
#         break
#
# # Webcam feed ended
# web_cam.release()
#
# # Close windows
# cv2.destroyAllWindows()
