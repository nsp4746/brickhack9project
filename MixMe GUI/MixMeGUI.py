import tkinter as tk
from tkinter import *
from tkinter import ttk

class MixMe(Frame):
    def __init__( self ):
        tk.Frame.__init__(self)
        self.pack()
        self.master.title("MixMe")
        self.button1 = Button( self, text = "CLICK HERE", width = 25,
                               command = self.new_window )
        self.button1.grid( row = 0, column = 1, columnspan = 2, sticky = W+E+N+S )
    def new_window(self):
        self.newWindow = MixMe2()
class MixMe2(Frame):     
    def __init__(self):
        new =tk.Frame.__init__(self)
        new = Toplevel(self)
        new.title("Coke")
        new.button = tk.Button(  text = "PRESS TO CLOSE", width = 25,
                                 command = self.close_window )
        new.button.pack()
    def close_window(self):
        self.destroy()
def main(): 
    MixMe().mainloop()
if __name__ == '__main__':
    main()