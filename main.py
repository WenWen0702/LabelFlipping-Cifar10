import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm_mean = [0.485, 0.456, 0.406]  # 均值
norm_std = [0.229, 0.224, 0.225]  # 方差

transform = transforms.Compose([transforms.ToTensor(),  # 将PILImage转换为张量
                                # 将[0,1]归一化到[-1,1]
                                transforms.Normalize(norm_mean, norm_std),
                                transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                ])

transform1 = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(norm_mean, norm_std)])

# 定义训练集trainset
trainset = torchvision.datasets.CIFAR10(root='../input/cifar10-python', train=True, download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='../input/cifar10-python', train=False, download=True, transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuralnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=False),
            nn.Dropout(0.4),
            nn.Flatten(),  # Flatten层
            nn.Linear(2048, 256),  # 全连接层
            nn.Linear(256, 64),  # 全连接层
            nn.Linear(64, 10)  # 全连接层
        )

    def forward(self, input):
        out = self.neuralnet(input)
        return out


net = Net().to(dev)

# 定义损失函数和优化器
# 导入torch.potim优化器模块
import torch.optim as optim

# 用了神经网络工具箱nn中的交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# optim模块中的SGD梯度优化方式---随机梯度下降
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# 训练
if __name__ == '__main__':
    for epoch in range(20):
        # 定义一个变量方便我们对loss进行输出
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = data
            # 把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()
            # 把数据输进网络net
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels.long())
            # loss进行反向传播
            loss.backward()
            # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
            optimizer.step()
            # 打印loss
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    detailer = iter(testloader)
    images, labels = detailer.next()

    # 展示这四张图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[int(labels[j])]:5s}' for j in range(128)))

    images = images.to(dev)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join(f'{classes[int(predicted[j])]:5s}' for j in range(128)))

    # 定义预测正确的图片数，初始化为0
    correct = 0
    # 总共参与测试的图片书，初始化为0
    total = 0
    # 循环每一个batch
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = images.to(dev), labels.to(dev)
            # 输入网络进行测试
            outputs = net(images)
            # 我们选择概率最高的类作为预测
            _, predicted = torch.max(outputs.data, 1)
            # 更新测试图片的数量
            total += labels.size(0)
            # 更新预测正确的数量
            correct += (predicted == labels).sum()
    # 输出结果
    print(f'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # 计算每一类的预测
    # 字典生成器
    correct_pred = {classname: 0 for classname in classes}
    # 创建包含10个类别的字典
    total_pred = {classname: 0 for classname in classes}
    # 以一个batch为单位进行循环
    for data in testloader:
        images, labels = images.to(dev), labels.to(dev)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # 收集每个类别的正确预测
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label.int()]] += 1
            total_pred[classes[label.int()]] += 1

    # 打印每一个类别的准确率
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
