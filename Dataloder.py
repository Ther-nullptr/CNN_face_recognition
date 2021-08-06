import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
from numpy.lib.shape_base import split
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
    trainLabelsOnehot = []
    
    validationImgs = []
    validationLabels = []
    validationLabelsOnehot = []

    testImgs = []
    testLabels = []
    testLabelsOnehot = []
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
        self.onehotLabel = np.eye(2)[self.labelList]
        # 读入图像数据
        
        count = 0
        length = len(self.nameList)
        start = time.time()
        print("begin loading pictures...")
        for name in tqdm(self.nameList):
            self.imgList.append(self.loadPicArray(".\\"+name))
            count += 1
        time_elapsed = time.time() - start
        print("done")
        print("{}s used".format(int(time_elapsed)))

        state = np.random.get_state()
        np.random.shuffle(self.imgList)
        np.random.set_state(state)
        np.random.shuffle(self.labelList)
        np.random.set_state(state)
        np.random.shuffle(self.onehotLabel)
        np.random.set_state(state)
        np.random.shuffle(self.nameList)

        self.trainImgs = self.imgList[0:self.trainIndex]
        self.trainLabels = self.labelList[0:self.trainIndex]
        self.trainLabelsOnehot = self.onehotLabel[0:self.trainIndex]

        self.validationImgs = self.imgList[self.trainIndex:self.validationIndex]
        self.validationLabels = self.labelList[self.trainIndex:self.validationIndex]
        self.validationLabelsOnehot = self.onehotLabel[self.trainIndex:self.validationIndex]

        self.testImgs = self.imgList[self.validationIndex:]
        self.testLabels = self.labelList[self.validationIndex:]
        self.testLabelsOnehot = self.onehotLabel[self.validationIndex:]
        self.testNames = self.nameList[self.validationIndex:]

        print("train images:{}, validate images:{}, test images:{}".format(len(self.trainImgs),len(self.validationImgs),len(self.testImgs)))
        
    def loadPicArray(self, picFilePath):
        picData = cv2.imread(picFilePath,flags=cv2.IMREAD_GRAYSCALE)
        newpicData = cv2.resize(picData,dsize=(50,50)) # 将图像划为50*50
        newpicData = np.expand_dims(newpicData/255,0) # 归一化并调整维度
        return newpicData

    def getTrainData(self): # 将array直接转换为tensor
        return torch.tensor(self.trainImgs), torch.tensor(self.trainLabels), torch.tensor(self.trainLabelsOnehot)

    def getValidationData(self):
        return torch.tensor(self.validationImgs), torch.tensor(self.validationLabels), torch.tensor(self.validationLabelsOnehot)

    def getTestData(self):
        return torch.tensor(self.testImgs), torch.tensor(self.testLabels), torch.tensor(self.testLabelsOnehot)
    
    def getTestNames(self):
        return self.testNames

    def printlist(self):
        print(self.testLabelsOnehot)

        '''
        self.fileslist = []
        for path,dirs,files in os.walk(file_dir, topdown=True):
            for file in files:
                self.fileslist.append(files)
            print(path)
            print(dirs)
            print(files)     
            # 当前目录下所有子目录   
            # self.dirname.append(dirs)
        '''



if __name__ == '__main__':
    dataloader = MyDataLoader()
    dataloader.printlist()
    
'''
    dataloader = DataLoader()
    filename = 'lfw_funneled\\Aaron_Guiel\\Aaron_Guiel_0001.jpg'
    img = cv2.imread(filename=filename,flags=cv2.IMREAD_COLOR)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''