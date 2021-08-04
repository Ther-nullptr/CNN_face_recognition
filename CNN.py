from torch import optim
import torch
from torch import nn
from Dataloder import MyDataLoader
import torch.utils.data as Data

torch.set_default_tensor_type(torch.DoubleTensor)

# 超参数的设定
batchSize = 200
lr = 0.001
momentum = 0
epoch = 0
inputNums = 50*50
outputNums = 2
mean = 0
std = 0.10
val = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def getLoader():
    dataloader = MyDataLoader()
    trainImgs, trainLabels, trainLabelsOnehot = dataloader.getTrainData()
    validationImgs, validationLabels, validationLabelsOnehot = dataloader.getValidationData()
    testImgs, testLabels, testLabelsOnehot = dataloader.getTestData()

    # 将数据特征和标签进行组合
    trainDataset = Data.TensorDataset(trainImgs,trainLabels)
    validationDataset = Data.TensorDataset(validationImgs,validationLabels)
    testDataset = Data.TensorDataset(testImgs,testLabels)

    # 构建dataloader()
    trainLoader = Data.DataLoader(dataset=trainDataset,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  num_workers=2)

    validationLoader = Data.DataLoader(dataset=validationDataset,
                                       batch_size=batchSize,
                                       shuffle=True,
                                       num_workers=2)

    testLoader = Data.DataLoader(dataset=testDataset,
                                 batch_size=batchSize,
                                 shuffle=True,
                                 num_workers=2)
    return trainLoader,validationLoader,testLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1) # 随机删除一些神经元
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3,3),
                stride=(1,1),
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear1 = nn.Linear(in_features=256,out_features=32)
        self.linear2 = nn.Linear(in_features=32,out_features=outputNums)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        temp = x.view(x.shape[0], -1)
        temp2 = self.linear1(temp)
        output = self.linear2(temp2)
        return output

def getModel():
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(),lr=lr,betas=(0.9,0.999),weight_decay=0)
    loss = nn.CrossEntropyLoss()
    return cnn,optimizer,loss

def train(cnn,optimizer,loss,trainLoader):
    for i in range(1,epoch+1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X,Y in trainLoader:
            Y_hat = cnn(X)
            l = loss(Y_hat,Y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (Y_hat.argmax(dim=1) == Y).sum().item()
            n+=Y.shape[0]
        print("epoch %d, loss %.4f, train accuracy %.1f%%"
			      % (i, train_l_sum / n, 100*train_acc_sum / n))

def test(cnn,loss,testLoader):
    test_l_sum,test_acc_sum, n = 0.0,0.0, 0
    with torch.no_grad(): # 关闭梯度记录
        for X, Y in testLoader:
            Y_hat = cnn(X)
            l = loss(Y_hat,Y)
            test_l_sum += l.item()
            test_acc_sum +=  (Y_hat.argmax(dim=1) == Y).sum().item()
            n += batchSize
        print(".......")
        print("test: loss %.4f, train accuracy %.1f%%"
                  %(test_l_sum/n,100*test_acc_sum/n))

if __name__ == "__main__":
    trainLoader,validationLoader,testLoader = getLoader()
    cnn,optimizer,loss = getModel()
    train(cnn,optimizer,loss,trainLoader)
    test(cnn,loss,testLoader)


'''
class Network:  # 一个简单的softmax分类网络
    def __init__(self):
        self.net = nn.Sequential(
            OrderedDict([('linear', nn.Linear(inputNums, outputNums))]))
        init.normal_(self.net.linear.weight, mean=mean, std=std)  # 设置初始权重
        init.constant_(self.net.linear.bias, val=val)  # 设置偏移
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=lr,
                                         momentum=momentum)  # 随机梯度下降法
'''
