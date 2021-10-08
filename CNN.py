from torch import optim
import torch
from torch import nn
from torchvision import transforms
import torch.utils.data as Data

from termcolor import *
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np

import time
import sys
import os

from Dataloder import MyDataLoader
from Dataset import MyDataset

# 默认设置
warnings.filterwarnings('ignore') # 关闭警告
torch.set_default_tensor_type(torch.DoubleTensor) # 设置默认数据类型
currentpath = os.path.abspath('.') # 获取当前路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数的设定
lr = 0.001
batchSize = 20
epoch = 30
lossRate = 0.5

# 命令行参数模式
if len(sys.argv)>1:
    lr = float(sys.argv[1])
    batchSize = int(sys.argv[2])
    epoch = int(sys.argv[3])
    lossRate = float(sys.argv[4])

# 记录训练状况，以备作图
train_acc = []
train_loss = []
validate_acc = []
validate_loss = []

# 获取数据，并对数据进行预处理
def getLoader():
    # 获取dataloader中的数据
    dataloader = MyDataLoader()
    trainImgs, trainLabels= dataloader.getTrainData()
    validationImgs, validationLabels= dataloader.getValidationData()
    testImgs, testLabels = dataloader.getTestData()
    testNames = dataloader.getTestNames()

    # 针对训练集的图片变换方法
    transform1 = transforms.Compose([
        transforms.Resize((50,50)), # resize图片
        transforms.RandomRotation((-20,20)), #随机旋转
        transforms.RandomHorizontalFlip(), #随机水平旋转
        transforms.ToTensor(), # 将图片信息转换为Tensor数组
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    # 针对验证集和测试集的变换方法
    transform2 = transforms.Compose([
        transforms.Resize((50,50)),
        transforms.ToTensor(), # 将图片信息转换为Tensor数组
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    # 将数据特征和标签进行组合,并进行处理
    trainData = []
    validationData = []
    testData = []
    for i in range(len(trainImgs)):
        trainData.append([trainImgs[i], trainLabels[i]])
    for i in range(len(validationImgs)):
        validationData.append([validationImgs[i], validationLabels[i]])
    trainDataset = MyDataset(trainData, transform1)
    validationDataset = MyDataset(validationData, transform2)

    # 构建dataloader()
    trainLoader = Data.DataLoader(dataset=trainDataset,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  num_workers=2)

    validationLoader = Data.DataLoader(dataset=validationDataset,
                                       batch_size=batchSize,
                                       shuffle=True,
                                       num_workers=2)

    for i in range(len(testImgs)):
        testData.append(transform2(testImgs[i]))
    testLoader = (testData, testLabels, testNames)

    return trainLoader, validationLoader, testLoader

# 神经网络定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)

        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=0)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=256, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # 对卷积层和线性层的权重执行 He 均匀分布初始化
                torch.nn.init.kaiming_normal_(
                    tensor=m.weight.data,  # 需要初始化的对象
                    a=0,  # LeakyReLU 负半轴斜率，默认为 0，即 ReLU
                    mode='fan_in',  # 可传入 'fan_in' 或 'fan_out'两种值，默认为 'fan_in'
                                    # 'fan_in' 表示设置标准差时，分母为输入层的神经元个数，正向传播时方差一致
                                    # 'fan_out' 表示设置标准差时，分母为该层神经元个数，反向传播时方差一致
                    nonlinearity='leaky_relu'  # 网络中使用的激活函数，默认为 'leaky_relu'
                )


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = x.view(x.shape[0], -1) # 将数据设置为1维
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x


# 获取训练所需的网络和梯度下降方法等
def getModel():
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           weight_decay=0)
    loss = nn.CrossEntropyLoss()
    cnn = cnn.to(device)
    return cnn, optimizer, loss

# 训练
def train(cnn, optimizer, loss, trainLoader):
    global train_acc,train_loss
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, Y in tqdm(trainLoader):
        X, Y = X.to(device), Y.to(device)
        Y_hat = cnn(X)
        l = loss(Y_hat, Y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train_l_sum += l.item()
        train_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
        n += Y.shape[0]
    print("train: loss %.5f, train accuracy %.1f%%" %
          (train_l_sum / n, 100 * train_acc_sum / n))
    train_loss.append(train_l_sum / n)
    train_acc.append(100*train_acc_sum/n)


# 验证
def validate(cnn ,loss, validateLoader):
    global validate_acc,validate_loss
    validate_l_sum, validate_acc_sum, n = 0.0, 0.0, 0
    with torch.no_grad():  # 关闭梯度记录
        for X, Y in tqdm(validateLoader):
            X, Y = X.to(device), Y.to(device)
            Y_hat = cnn(X)
            l = loss(Y_hat, Y)
            validate_l_sum += l.item()
            validate_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n += Y.shape[0]
        print("validate: loss %.5f, validate accuracy %.1f%%" %
              (validate_l_sum / n, 100 * validate_acc_sum / n))
        print("..........................................................................")
    validate_loss.append(validate_l_sum / n)
    validate_acc.append(100 * validate_acc_sum / n)


# 测试
def test(cnn,testLoader):
    test_acc_sum, n = 0.0, 0
    #model = torch.load('cnn.pkl')
    with torch.no_grad():  # 关闭梯度记录
        for i in range(len(testLoader[0])):
            X = testLoader[0][i].unsqueeze(0)
            Y = testLoader[1][i]
            X,Y = X.to(device),Y.to(device)
            name = testLoader[2][i]
            actualSex = ''
            predictSex = ''
            color = 'green'
            Y_hat = cnn(X)
            test_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n += 1
            if (i % 100 == 0):
                if (Y == 0):
                    actualSex = 'female'
                else:
                    actualSex = 'male'

                if (Y_hat.argmax(dim=1) == 0):
                    predictSex = 'female'
                else:
                    predictSex = 'male'

                if (Y_hat.argmax(dim=1) == Y):
                    color = 'green'
                else:
                    color = 'red'
                print(currentpath + '\\' + name)
                print(
                    colored(
                        "actual:{}\tpredicted:{}".format(
                            actualSex, predictSex), color))
        print("test accuracy %.1f%%" % (100 * test_acc_sum / n))


# 主函数
if __name__ == "__main__":
    trainLoader, validationLoader, testLoader = getLoader()
    cnn, optimizer, loss = getModel()
    dirname = 'lr={},batch={},epoch={},lr_loss={}'.format(lr,batchSize,epoch,lossRate)
    logfile = 'log.txt'
    with open(logfile,'a') as f:
        f.write(dirname)
        f.write('\n')
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass
    start = time.time()
    
    for i in range(1, epoch + 1):
        print("epoch {}:".format(i))
        cnn.train()
        train(cnn, optimizer, loss, trainLoader)
        cnn.eval()
        validate(cnn, loss, validationLoader)
        if i%5 == 0:
            lr = lr * lossRate
        if i == epoch:
            torch.save(cnn,'cnn.pkl')

    time_elapsed = time.time() - start
    print("done")
    print("{}s used".format(int(time_elapsed)))

    x = np.arange(1,epoch+1,1)
    plt.figure(1)
    plt.plot(x, train_acc, label='train accuracy')
    plt.plot(x, validate_acc, label='validate accuracy')
    plt.legend()
    plt.savefig('./'+dirname+'/accuracy.jpg')

    plt.figure(2)
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, validate_loss, label='validate loss')
    plt.legend()
    plt.savefig('./'+dirname+'/loss.jpg')
    
    cnn.eval()
    test(cnn,testLoader)
