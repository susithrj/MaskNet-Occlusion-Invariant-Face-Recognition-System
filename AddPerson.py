# Importing the libraries
import cv2
import os
from tkinter import *

# Variables
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
l_1.grid(row=1, column=col_num, sticky='NSEW')

# Inputs
e1_val = StringVar()
e_1 = Entry(main_frame, textvariable=e1_val, bg=bg_color, fg=fg_color)
e_1.grid(row=1, column=col_num + 1)

# UI functions
def pass_inputs():
    return e1_val.get()

def complete_information_gathering():
    main_frame.destroy()

# Buttons
b1 = Button(main_frame, text="OK", bg=bg_color, fg=fg_color, command=complete_information_gathering)
b1.grid(row=2, column=col_num+1)

main_frame.mainloop()
# End of UI functions

'''
Function for creating directory to collect data.
'''

def assure_folder_exists(folder):
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


#creating directory
nametest = pass_inputs()
print("created directory for name: "+nametest)
#making directory for new person
os.makedirs('data_captures/'+nametest)
assure_folder_exists('data_captures/'+ nametest)


'''
Capturing faces from camera.
'''
# Load Haarcascades
face_classifier = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')


def face_extractor(img):

    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


# Initialize camera
cap = cv2.VideoCapture(0)
count = 0
# Collect samples
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))

        # Save files to created dir
        file_name_path = 'data_captures/'+nametest+'/' + str(count) + '.jpg'
        # file_name_path = 'data_captures/1/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # display live count
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
