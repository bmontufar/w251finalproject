import tkinter as tk
import subprocess
from tkinter import *
from PIL import Image, ImageTk
import os
root = tk.Tk()
root.geometry("600x500")


#cmd = 'OPENBLAS_CORETYPE=ARMV8 python3 skull_tattoo_gen.py'
#os.system(cmd)

#cmd = 'OPENBLAS_CORETYPE=ARMV8 python3 scorpion_tattoo_gen.py'
#os.system(cmd)
def run_tattoo_gen():
    cmd = 'OPENBLAS_CORETYPE=ARMV8 python3 bird_tattoo_gen.py'
    os.system(cmd)

def run_opencv():
    subprocess.call(["python3", "tattoo_placement.py"])



def bird_gen():
    canvas = Canvas(root, height=128, width=128)
    canvas.pack()
    my_image = PhotoImage(file='bird.png', master= root)
    canvas.create_image(0, 0, anchor=NW, image=my_image)
    canvas.place(x=236, y=150)
    with open('tattoo.txt', 'w') as f:
    	f.write('bird')

    root.mainloop()

def skull_gen():
    canvas = Canvas(root, height=128, width=128)
    canvas.pack()
    my_image = PhotoImage(file='skull.png', master= root)
    canvas.create_image(0, 0, anchor=NW, image=my_image)
    canvas.place(x=236, y=150)
    with open('tattoo.txt', 'w') as f:
    	f.write('skull')
    root.mainloop()

def scorpion_gen():
    canvas = Canvas(root, height=128, width=128)
    canvas.pack()
    my_image = PhotoImage(file='scorpion.png', master= root)
    canvas.create_image(0, 0, anchor=NW, image=my_image)
    canvas.place(x=236, y=150)
    with open('tattoo.txt', 'w') as f:
    	f.write('scorpion')
    root.mainloop()

run_tattoo_gen()

# add background img
bg = PhotoImage(file = "bg.png")
label1 = Label( root, image = bg)
label1.place(x = 0, y = 0)
# button quit
button = tk.Button(
                   text="QUIT",
                   fg="red",
                   width = 10,
                   height = 2,
                   command=quit)
button.place(x=450, y=400)

# button regen
button0 = tk.Button(
                   text="REGEN",
                   width = 10,
                   height = 2,
                   command=run_tattoo_gen)
button0.place(x=450, y=40)


# button tattoo fit
button1 = tk.Button(
                   width = 10,
                   height = 2,
                   text="Fit Tattoo!",
                   command=run_opencv)
button1.place(x=350, y=400)


# button generate bird tattoo
button2 = tk.Button( width = 10, height = 2, text="Bird",command=bird_gen)
button2.place(x=250, y=400)



# button generate scorpion tattoo
button2 = tk.Button(
                   width = 10,
                   height = 2,
                   text="Scorpion",
                   command=scorpion_gen)
button2.place(x=150, y=400)


# button generate skull tattoo
button2 = tk.Button(
                   width = 10,
                   height = 2,
                   text="Skull",
                   command=skull_gen)
button2.place(x=50, y=400)



# main
root.mainloop()
