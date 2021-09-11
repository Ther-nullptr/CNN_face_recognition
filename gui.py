import tkinter
from tkinter.constants import *
from tkinter import messagebox
import tkinter.font as tkFont

from PIL import Image, ImageTk

import torch
from Dataloder import MyDataLoader
from torchvision import transforms

import re
import os

from CNN import CNN
currentpath = os.path.abspath('.')


class App:
    def __init__(self): # gui组件
        # 后台运行
        self.LoadResource()

        # 图片指标
        self.index = 0

        # 基本窗口
        self.top = tkinter.Tk()
        self.top.title("face recognition system")  # 创建窗口
        self.top.geometry('500x300')  # 设置窗口大小,注意中间是x

        # 图片组件
        self.img = Image.open(self.testNames[self.index])
        self.img = self.img.resize((170, 170))
        self.photo = ImageTk.PhotoImage(self.img)
        self.imgLabel = tkinter.Label(self.top, image=self.photo)  #把图片整合到标签类中
        self.imgLabel.place(x=180, y=25)  #自动对齐

        # 文字信息(静态)
        self.name_label = tkinter.Label(self.top,
                                        text="name:",
                                        font=("Cascadia Code", 10,
                                              tkFont.BOLD))
        self.name_label.place(x=20, y=20)

        self.predicted_label = tkinter.Label(self.top,
                                             text="predicted gender:",
                                             font=("Cascadia Code", 10,
                                                   tkFont.BOLD))
        self.predicted_label.place(x=20, y=60)

        self.actual_label = tkinter.Label(self.top,
                                          text="actual gender:",
                                          font=("Cascadia Code", 10,
                                                tkFont.BOLD))
        self.actual_label.place(x=20, y=100)

        self.result_label = tkinter.Label(self.top,
                                          text="result:",
                                          font=("Cascadia Code", 10,
                                                tkFont.BOLD))
        self.result_label.place(x=20, y=140)

        # 文字信息(动态)
        self.name = tkinter.Label(self.top,
                                  text=self.names[self.index],
                                  font=("Consolas", 10))
        self.name.place(x=20, y=40)

        self.predictedGender = tkinter.Label(self.top,
                                             text="",
                                             font=("Consolas", 10))
        self.predictedGender.place(x=20, y=80)

        self.actualGender = tkinter.Label(self.top,
                                          text="",
                                          font=("Consolas", 10))
        self.actualGender.place(x=20, y=120)

        self.result = tkinter.Label(self.top,
                                    text="", 
                                    font=("Consolas", 10))
        self.result.place(x=20, y=160)

        # 按钮信息

        self.last_Button = tkinter.Button(self.top,
                                          text="last",
                                          command=self.Last,
                                          font=("Consolas", 10))
        self.last_Button.place(x=400, y=70)

        self.next_Button = tkinter.Button(self.top,
                                          text="next",
                                          command=self.Next,
                                          font=("Consolas", 10))
        self.next_Button.place(x=400, y=100)

        self.B1 = tkinter.Button(self.top,
                                 text="test one",
                                 command=self.Testone,
                                 font=("Consolas", 10))
        self.B1.place(x=200, y=240)

        self.B2 = tkinter.Button(self.top,
                                 text="test all",
                                 command=self.Testall,
                                 font=("Consolas", 10))
        self.B2.place(x=270, y=240)

    # 加载资源
    def LoadResource(self): 
        # 加载网络和原始数据
        self.net = torch.load('cnn.pkl')
        self.dataLoader = MyDataLoader()
        self.testImgs, self.testLabels = self.dataLoader.getTestData()

        # 图片变换
        transform = transforms.Compose([
        transforms.Resize((50,50)),
        transforms.ToTensor(), # 将图片信息转换为Tensor数组
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        self.testData = []
        for i in range(len(self.testImgs)):
            self.testData.append(transform(self.testImgs[i]))

        # 提取名字
        self.testNames = self.dataLoader.getTestNames()
        self.names = []
        for item in self.testNames:
            name = re.findall(r".*\\(.*)\\.*", item)[0]
            self.names.append(name)

    # 测试所有图片
    def Testall(self): 
        test_acc_sum = 0
        with torch.no_grad():  # 关闭梯度记录
            for i in range(len(self.testNames)):
                X = self.testData[i].unsqueeze(0)
                Y = self.testLabels[i]
                Y_hat = self.net(X)
                test_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
        messagebox.showinfo(
            "result", "train accuracy %.1f%%" %
            (100 * test_acc_sum / len(self.testNames)))

    # 测试单张图片
    def Testone(self):
        X = self.testData[self.index].unsqueeze(0)
        Y = self.testLabels[self.index]
        color = 'green'
        _result = ''
        Y_hat = self.net(X)
        if (Y_hat.argmax(dim=1) == Y):
            color = 'green'
            _result = 'right'
        else:
            color = 'red'
            _result = 'wrong'
        self.predictedGender.configure(text=self.getGender(Y_hat.argmax(
            dim=1)))
        self.actualGender.configure(text=self.getGender(Y))
        self.result.configure(text=_result,fg=color)

    # 切换为上一张图片
    def Last(self):
        self.index -= 1
        self.img = Image.open(self.testNames[self.index])
        self.img = self.img.resize((170, 170))
        self.photo = ImageTk.PhotoImage(self.img)
        self.imgLabel.configure(image=self.photo)
        self.name.configure(text=self.names[self.index])
        self.predictedGender.configure(text='')
        self.actualGender.configure(text='')
        self.result.configure(text='')

    # 切换为下一张图片
    def Next(self):
        self.index += 1
        self.img = Image.open(self.testNames[self.index])
        self.img = self.img.resize((170, 170))
        self.photo = ImageTk.PhotoImage(self.img)
        self.imgLabel.configure(image=self.photo)
        self.name.configure(text=self.names[self.index])
        self.predictedGender.configure(text='')
        self.actualGender.configure(text='')
        self.result.configure(text='')

    # 获取性别
    def getGender(self, num):
        if num == 0:
            return "female"
        else:
            return "male"

    # 主运行程序
    def Run(self):
        self.top.mainloop()

if __name__ == '__main__':
    app = App()
    app.Run()
