from torch import optim
import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import linear
from Dataloder import MyDataLoader
import torch.utils.data as Data
from termcolor import *
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
is_train = True  # 是否先训练再测试
currentpath = os.path.abspath('.')

# 超参数的设定
batchSize = 200
lr = 0.002
momentum = 0
epoch = 30
inputNums = 50 * 50
outputNums = 2
mean = 0
std = 0.10
val = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getLoader():
    dataloader = MyDataLoader()
    trainImgs, trainLabels, trainLabelsOnehot = dataloader.getTrainData()
    validationImgs, validationLabels, validationLabelsOnehot = dataloader.getValidationData(
    )
    testImgs, testLabels, testLabelsOnehot = dataloader.getTestData()
    testNames = dataloader.getTestNames()

    # 将数据特征和标签进行组合
    trainDataset = Data.TensorDataset(trainImgs, trainLabels)
    validationDataset = Data.TensorDataset(validationImgs, validationLabels)

    # 构建dataloader()
    trainLoader = Data.DataLoader(dataset=trainDataset,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  num_workers=2)

    validationLoader = Data.DataLoader(dataset=validationDataset,
                                       batch_size=batchSize,
                                       shuffle=True,
                                       num_workers=2)

    testLoader = (testImgs, testLabels, testNames)

    return trainLoader, validationLoader, testLoader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
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

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=256, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=outputNums)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):  # 对卷积层和线性层的权重执行 Xavier 均匀分布初始化
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
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x


def getModel():
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           weight_decay=0)
    loss = nn.CrossEntropyLoss()
    return cnn, optimizer, loss


def train(cnn, optimizer, loss, trainLoader):
    for i in range(1, epoch + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, Y in tqdm(trainLoader):
            Y_hat = cnn(X)
            l = loss(Y_hat, Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n += Y.shape[0]
        print(
            ".........................................................................."
        )
        print("epoch %d, loss %.5f, train accuracy %.1f%%" %
              (i, train_l_sum / n, 100 * train_acc_sum / n))
        torch.save(cnn, 'cnn.pkl')
        print(
            ".........................................................................."
        )


def validate(loss, validateLoader):
    validate_l_sum, validate_acc_sum, n = 0.0, 0.0, 0
    model = torch.load('cnn.pkl')
    with torch.no_grad():  # 关闭梯度记录
        for X, Y in validateLoader:
            Y_hat = model(X)
            l = loss(Y_hat, Y)
            validate_l_sum += l.item()
            validate_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n += Y.shape[0]
        print(
            ".........................................................................."
        )
        print("validate: loss %.5f, validate accuracy %.1f%%" %
              (validate_l_sum / n, 100 * validate_acc_sum / n))
        print(
            ".........................................................................."
        )


def test(testLoader):
    test_acc_sum, n = 0.0, 0
    model = torch.load('cnn.pkl')
    with torch.no_grad():  # 关闭梯度记录
        for i in range(len(testLoader[0])):
            X = testLoader[0][i].unsqueeze(0)
            Y = testLoader[1][i]
            name = testLoader[2][i]
            actualSex = ''
            predictSex = ''
            color = 'green'
            Y_hat = model(X)
            test_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()

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

            n += 1
        print(
            ".........................................................................."
        )
        print("train accuracy %.1f%%" % (100 * test_acc_sum / n))
        print(
            ".........................................................................."
        )


if __name__ == "__main__":
    trainLoader, validationLoader, testLoader = getLoader()
    cnn, optimizer, loss = getModel()
    train(cnn, optimizer, loss, trainLoader)
    validate(loss, validationLoader)
    test(testLoader)