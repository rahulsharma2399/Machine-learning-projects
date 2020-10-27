import tkinter as tk
from  beng import *
from PIL import ImageTk,Image

fields = ('Enter Location', 'sqft', 'number of bathroom', 'number of kitchen',"Price" )

print(predict_price('Indira Nagar',1000, 3, 3))
def final_balance(entries):

    loc = entries['Enter Location'].get()

    sq = float(entries['sqft'].get())
    ba =  int(entries['number of bathroom'].get()) 
    kit = float(entries['number of kitchen'].get())
    p=predict_price(loc,sq,ba,kit)
    entries['Price'].insert(0, p)
    

def makeform(root, fields):
    entries = {}
    i=100
    j=200
    for field in fields:
        print(field)
        row = tk.Frame(root)
        lab = tk.Label(row, width=22, text=field+": ", anchor='w')
        ent = tk.Entry(row)
        
        row.pack(side=tk.TOP, 
                 fill=tk.X, 
                 padx=5, 
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, 
                 expand=tk.YES, 
                 fill=tk.X)
        entries[field] = ent
    return entries

if __name__ == '__main__':
    root = tk.Tk()
    ents = makeform(root, fields)
    b1 = tk.Button(root, text='Click to find estimated price',
           command=(lambda e=ents: final_balance(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)

    
    background_image=tk.PhotoImage("images.jpeg")
    background_label = tk.Label(root, image=background_image) 
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    root.mainloop()
