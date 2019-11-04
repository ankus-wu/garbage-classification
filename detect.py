
# coding: utf-8

# In[125]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

import os
from PIL import Image,ImageTk
import PIL.Image as immm
classes = ['cardboard','glass','metal','paper','plastic','trash']


# In[162]:



def findfile():
    #frame = tkinter.Toplevel()
    filepath = askopenfilename(title='Select the file', filetypes=[('All Files', '*')],initialdir='./')#C:\\Windows
    img = immm.open(filepath)
    img_png = ImageTk.PhotoImage(img)
    label1.config(image=img_png,bg = 'grey',height=400,width=400)
    #label1.image=img_png
    #Image2 = canvas.create_image(250, 0, anchor='n',image=img_png)  
    #Image2 = tkinter.Label(mygui,bg = 'white',bd =20,height=110,width=110,image =img_png)
    #Image2.pack()
    
    trans = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    img = trans(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    if os.path.exists('save_resmodel.pkl'):
        model = torch.load('save_resmodel.pkl').cuda()
    model.eval()
    pred = model(img)
    _, predicted = torch.max(pred, 1)
    result = classes[predicted]
    
    label2['text']='该垃圾的类别为 '+result
    
    mygui.mainloop()


# In[163]:


from tkinter import * 
from tkinter.filedialog import askopenfilename
# 创建Tk对象，Tk代表窗口
mygui = Tk(className='分类')
mygui.geometry('600x600')
label =Label(mygui)
label['text']='打开一张图片即可自动检测'
label.pack()
text=StringVar()
#text.set('wait')
label1 = Label(mygui)
label1.pack()
# 设置窗口标题
mymenu=Menu()
mymenu.add_command(label='打开',command=lambda:findfile())


# 创建Label对象，第一个参数指定该Label放入root


label2 =Label(mygui)
label2.pack()

'''entry = Entry(mygui)
entry['textvariable']=text
entry.pack()'''
'''
btn = Button()
btn['text'] = '检测'
btn['command'] = lambda:read_image(filepath) 
btn.pack()'''
# 启动主窗口的消息循环
mygui.config(menu=mymenu)
mygui.mainloop()

