'''一个面向过程风格的线性回归训练模型'''
import numpy as np
import torch
from torch.optim import optimizer
import torch.utils.data as Data
from torch import nn # pytorch中实现神经网络的模块 
from torch.nn import init
from torch import optim
from collections import OrderedDict


torch.set_default_tensor_type(torch.FloatTensor) # 几个设定
torch.manual_seed(1)


# 简易的手工指定,放在类外
num_inputs = 2
num_examples = 1000
batch_size = 10 # 每批样本的大小
file_path = '' 
lr = 0.01 # 学习率
num_epochs = 100 # epoch是指,所有训练样本在神经网络中都 进行了一次正向传播 和一次反向传播 .这里指定了循环的次数
true_w = [2, -3.4]
true_b = 4.2 # 验证用数据,在此我们先单独列出

def GetData():
    num_inputs = 2
    num_examples = 1000
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    print(features)
    print(labels)
    print(features.shape)
    print(labels.shape)


    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 把 dataset 放入 DataLoader
    data_iter = Data.DataLoader(
    dataset=dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
    )
    return data_iter

def GetNetwork():
    # 1.建层
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(num_inputs, 1)) # 添加线性层
    # 2.初始化
    init.normal_(net[0].weight,mean = 0,std = 0.01) # 初始化权重
    init.constant_(net[0].bias,val = 0) # 初始化偏差
    # 3.定义损失函数
    loss = nn.MSELoss() # 此处采用了均方误差
    # 4.定义优化算法
    optimizer = optim.SGD(net.parameters(),lr = lr) # 学习率;parameters()指这个学习率用于所有参数
    return net,loss,optimizer # 神经网络暴露在外的接口

def TrainNetwork(data_iter,net,loss,optimizer):
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter: # X每行有两个数据
            output = net(X) # 将x作为神经网络的输入
            l = loss(output, y.view(-1, 1)) # 将y的维度变为1列,-1表示行自适应
            optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
            l.backward() # 后向传播便于求导
            optimizer.step() # 按照学习率的设定移动一小步(梯度的信息已经全部被保存,所以不需要再进行传参)
        print('epoch %d, loss: %f' % (epoch, l.item()))

def TestNetwork(dense): # 测试的是网络层的参数
    print(true_w,dense.weight.data)
    print(true_b, dense.bias.data)

if __name__ == '__main__':
    data_iter = GetData()
    net,loss,optimizer = GetNetwork()
    TrainNetwork(data_iter=data_iter,net=net,loss=loss,optimizer=optimizer)
    TestNetwork(net[0])

# 1.传统的设置神经网络设置
'''
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__() # 初始化父类
        self.linear = nn.Linear(n_feature, 1) # 构造输入维数和输出维数
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y
'''
# 还可以直接用nn.Sequential搭建神经网络层,甚至可以给神经网络层命名

# 2.分层指定学习率
'''
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
'''


# net.parameters()