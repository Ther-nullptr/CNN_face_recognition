import numpy as np
from PIL import Image
import time
import torch
from tqdm import tqdm

class MyDataLoader: 
    # 加载LFW所有的数据集
    nameList = []
    imgList = []
    labelList = []
    onehotLabel = []
    
    # 分割后的数据集
    trainImgs = []
    trainLabels = []
    
    validationImgs = []
    validationLabels = []
    
    testImgs = []
    testLabels = []
    testNames = []

    def __init__(self):
        # 记录文件名称
        with open("female_names.txt") as f1:
            for name in f1:
                self.nameList.append("lfw_funneled"+"\\" + name.strip()[0:-9] + "\\" + name.strip()) # 根据数据集的特点获取文件名称
                self.labelList.append(0)
        del self.nameList[-1] # 删除文件末尾的换行符
        del self.labelList[-1] 
        with open("male_names.txt") as f2:
            for name in f2:
                self.nameList.append("lfw_funneled"+"\\" + name.strip()[0:-9] + "\\" + name.strip())
                self.labelList.append(1)
        del self.nameList[-1] 
        del self.labelList[-1] 
        self.dataCount = len(self.labelList)
        self.trainIndex = int(self.dataCount*0.4)
        self.validationIndex = int(self.dataCount*0.5)

        # 读入图像数据
        start = time.time()
        print("begin loading pictures...")
        count = 0
        for name in tqdm(self.nameList):
            self.imgList.append(self.loadPicArray(".\\"+name))
            count += 1
        time_elapsed = time.time() - start
        print("done")
        print("{}s used".format(int(time_elapsed)))

        # 打乱数据
        state = np.random.get_state()
        np.random.shuffle(self.imgList)
        np.random.set_state(state)
        np.random.shuffle(self.labelList)
        np.random.set_state(state)
        np.random.shuffle(self.nameList)

        # 分割数据
        self.trainImgs = self.imgList[0:self.trainIndex]
        self.trainLabels = self.labelList[0:self.trainIndex]

        self.validationImgs = self.imgList[self.trainIndex:self.validationIndex]
        self.validationLabels = self.labelList[self.trainIndex:self.validationIndex]

        self.testImgs = self.imgList[self.validationIndex:]
        self.testLabels = self.labelList[self.validationIndex:]
        self.testNames = self.nameList[self.validationIndex:]

        print("train images:{}, validate images:{}, test images:{}".format(len(self.trainImgs),len(self.validationImgs),len(self.testImgs)))
        
    # 载入数据
    def loadPicArray(self, picFilePath):
        return Image.open(picFilePath).convert('RGB')

    # 获取数据
    def getTrainData(self): # 将array直接转换为tensor
        return self.trainImgs, torch.tensor(self.trainLabels)

    def getValidationData(self):
        return self.validationImgs, torch.tensor(self.validationLabels)

    def getTestData(self):
        return self.testImgs, torch.tensor(self.testLabels)
    
    def getTestNames(self):
        return self.testNames

