# import statements
from tkinter import *
import os
from datetime import datetime

# UI Elements
main_frame = Tk()

main_frame.configure(background="#C6E6FF")

# User Interface functions

def train_and_recognise_model():
    os.system("py 2_face_recognizer.py")

def attend():
    #os.startfile(os.getcwd() + "reports/" + str(datetime.now().date()) + '.xls')
    os.startfile(os.getcwd() +'/attendence.csv')
    pass

def add_person():
    os.system("py 1_add_person.py")

def delete_persons():
    os.system("py 4_delete_users.py")

def help_window():
    os.system("py 3_help_window.py")

def exit_program():
    main_frame.destroy()

# Class variables
padding = 5
news_sticky = N + E + W + S
font_size = 20
foreground_color = 'white'
top_background_color = 'dark blue'
font_type = 'open sans'
view_bg_color = "#6898BB"
exit_color = "#D88383"

# Frame title
main_frame.title("Facial Recognition System from MaskNet")

# Labels in the UI
Label(main_frame, text="MaskNet RECOGNITION SYSTEM\nResult of Research by Susith, Achala", font=(font_type, font_size), fg=foreground_color, bg=top_background_color, height=2).grid(row=0,
                                                                                                               column=0,
                                                                                                               sticky=news_sticky,
                                                                                                               padx=padding, pady=padding)

# Buttons


Button(main_frame, text="Recognize Faces", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=train_and_recognise_model)\
    .grid(row=1, column=0, sticky=news_sticky, padx=padding, pady=padding)



Button(main_frame, text="Attendance Sheet", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=attend).grid(row=2,
                                                                                                                 column=0,
                                                                                                                 sticky=news_sticky,
                                                                                                                 padx=padding,
                                                                                                                 pady=padding)

Button(main_frame, text="Add Person", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=add_person).grid(row=3,
                                                                                                                  column=0,
                                                                                                                  sticky=news_sticky,
                                                                                                                  padx=padding,
                                                                                                                  pady=padding)

Button(main_frame, text="Train Model", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=train_and_recognise_model)\
    .grid(row=4, column=0, sticky=news_sticky, padx=padding, pady=padding)

#
# Button(main_frame, text="Delete Person", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=delete_persons).grid(row=5,
#                                                                                                                              column=0,
#                                                                                                                              sticky=news_sticky,
#                                                                                                                              padx=padding,
#                                                                                                                              pady=padding)

Button(main_frame, text="Help", font=(font_type, font_size), bg=view_bg_color, fg=foreground_color, command=help_window).grid(row=6,
                                                                                                                             column=0,
                                                                                                                             sticky=news_sticky,
                                                                                                                             padx=padding,
                                                                                                                             pady=padding)

Button(main_frame, text="Exit", font=(font_type, font_size), bg=exit_color, fg=top_background_color, command=exit_program).grid(row=7,
                                                                                                                  padx=padding, pady=padding)

main_frame.mainloop()
