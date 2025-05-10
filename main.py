import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('txxxxxx.h5')

#dictionary to label all traffic signs class.
classes = {
    0: 'bus', 1: 'car', 2: 'truck', 3: 'van'
}
                 
# initial UI 
top=tk.Tk() 
top.geometry('800x600') 
top.title('Nhận dạng phương tiện giao thông ')
top.configure(background='#ffffff')

label=Label(top,background='#ffffff', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path) # mở file 
    image = image.resize((30,30)) # resize giảm kích thước 
    image = numpy.expand_dims(image, axis=0) 
    image = numpy.array(image)
# predict classes
    pred_probabilities = model.predict(image)[0] 
    pred = pred_probabilities.argmax(axis=-1)
    print('pos:::::', pred)
    sign = classes[pred]
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Nhận dạng",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#c71b20', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.9,rely=0.3)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        # uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5) # initial button for upload img  
upload.configure(background='#c71b20', foreground='white',font=('arial',10,'bold')) # custom button UI 

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

# lable
heading = Label(top, text="Nhận dạng phương tiện giao thông",pady=10, font=('arial',20,'bold'))
heading.configure(background='#ffffff',foreground='#364156')

heading1 = Label(top, text="Môn Học: Nhập môn trí tuệ nhân tạo",pady=10, font=('arial',20,'bold'))
heading1.configure(background='#ffffff',foreground='#364156')

heading2 = Label(top, text="Danh sách thành viên nhóm",pady=5, font=('arial',20,'bold'))
heading2.configure(background='#ffffff',foreground='#364156')

heading3 = Label(
    top,
    text="BÙI ĐÌNH TUẤN MSSV: N22DCCN095\n"
         "NGUYỄN TRƯƠNG DUY PHƯƠNG MSSV: N22DCCN062\n"
         "NGUYỄN THÀNH PHONG MSSV: N22DCCN059\n"
         "NGUYỄN NHẬT THI MSSV: N22DCCN080",
    pady=5,
    font=('arial', 20, 'bold'),
    background='#ffffff',
    foreground='#364156'
)
heading3.configure(background='#ffffff',foreground='#364156')

heading.pack()
heading1.pack()
heading2.pack()
heading3.pack()
top.mainloop() # display window
