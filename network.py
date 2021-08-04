from d2lzh_pytorch.utils import train
from numpy.random import shuffle
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

from torch.utils.data import dataset
from Dataloder import MyDataLoader
import torch.utils.data as Data
import time

torch.set_default_tensor_type(torch.DoubleTensor)

# 超参数的设定
batchSize = 200
lr = 0.0005
momentum = 0
epoch = 30
inputNums = 50*50
outputNums = 2
mean = 0
std = 0.01
val = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Network:  # 一个简单的softmax分类网络
    def __init__(self):
        self.net = nn.Sequential(
            OrderedDict([('linear', nn.Linear(inputNums, outputNums))]))
        init.normal_(self.net.linear.weight, mean=mean, std=std)  # 设置初始权重
        init.constant_(self.net.linear.bias, val=val)  # 设置偏移
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                         lr=lr,
                                         )  # 随机梯度下降法

    def getData(self):
    # 载入数据
        dataloader = MyDataLoader()
        trainImgs, trainLabels, trainLabelsOnehot = dataloader.getTrainData()
        validationImgs, validationLabels, validationLabelsOnehot = dataloader.getValidationData()
        testImgs, testLabels, testLabelsOnehot = dataloader.getTestData()

    # 将数据特征和标签进行组合
        trainDataset = Data.TensorDataset(trainImgs,trainLabels)
        validationDataset = Data.TensorDataset(validationImgs,validationLabels)
        testDataset = Data.TensorDataset(testImgs,testLabels)


    # 构建dataloader()
        self.trainLoader = Data.DataLoader(dataset=trainDataset,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  num_workers=4)

        self.validationLoader = Data.DataLoader(dataset=validationDataset,
                                       batch_size=batchSize,
                                       shuffle=True,
                                       num_workers=4)

        self.testLoader = Data.DataLoader(dataset=testDataset,
                                 batch_size=batchSize,
                                 shuffle=True,
                                 num_workers=4)
    def Train(self):
        for i in range (1,epoch+1):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0 # 训练损失和训练精确度
            for X,Y in self.trainLoader:
                Y_hat = self.net(X)
                l = self.loss(Y_hat,Y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                train_l_sum += l.item()
                
                train_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
                n+=batchSize

            print("epoch %d, loss %.4f, train accuracy %.1f%%"
			      % (i, train_l_sum / n, 100*train_acc_sum / n))

    def Test(self):
        test_l_sum,test_acc_sum, n = 0.0,0.0, 0
        with torch.no_grad(): # 关闭梯度记录
            for X, Y in self.testLoader:
                Y_hat = self.net(X)
                l = self.loss(Y_hat,Y)
                test_l_sum += l.item()
                test_acc_sum +=  (Y_hat.argmax(dim=1) == Y).sum().item()
                n += batchSize
            print(".......")
            print("test: loss %.4f, train accuracy %.1f%%"
                  %(test_l_sum/n,100*test_acc_sum/n))


if __name__ == '__main__':
    network = Network()
    network.getData()
    network.Train()
    network.Test()