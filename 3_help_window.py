from tkinter import *

# UI Elements
main_frame = Tk()
main_frame.title("Facial Recognition System")
main_frame.configure(bg='#A4CCD0')

descrip = "This facial recognition system is simple" \
          " to get up and running\n we can add users and register them into the system\n we can train users."

# Labels
var_1 = StringVar()
l_1 = Label(main_frame, textvariable=var_1, bg='black', fg='white', relief=RAISED)
var_1.set(descrip)
l_1.grid(row=1, column=1, sticky=N + E + W + S)

main_frame.mainloop()