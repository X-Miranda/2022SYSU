from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from PIL import Image


# 数据预处理
def trans():
    return transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Resize((256, 256)),  # 将图像大小调整为256x256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化处理
    ])


# 自定义数据集类用于加载测试数据
class get_dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # 获取图像文件的文件名（不带扩展名）
        self.img_labels = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # 获取图像文件的完整路径
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        # 返回图像文件的数量
        return len(self.img_files)

    def __getitem__(self, idx):
        # 根据索引获取图像路径
        img_path = self.img_files[idx]
        # 打开图像并转换为RGB格式
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            # 如果有预处理变换，则应用变换
            img = self.transform(img)
        # 根据文件名获取标签
        label = self.get_label_from_filename(self.img_labels[idx])
        return img, label

    def get_label_from_filename(self, filename):
        # 根据文件名分配标签
        if 'baihe' in filename:
            return 0
        elif 'dangshen' in filename:
            return 1
        elif 'gouqi' in filename:
            return 2
        elif 'huaihua' in filename:
            return 3
        elif 'jinyinhua' in filename:
            return 4
        else:
            raise ValueError(f"Unknown: {filename}")


# 数据加载
def load_data(data_dir):
    # 获取数据预处理方法
    transform = trans()
    # 加载训练集
    train_d = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    # 加载测试集
    test_d = get_dataset(img_dir=f'{data_dir}/test', transform=transform)

    # 创建训练数据加载器
    train_l = DataLoader(train_d, batch_size=32, shuffle=True, num_workers=4)
    # 创建测试数据加载器
    test_l = DataLoader(test_d, batch_size=32, shuffle=False, num_workers=4)

    return train_l, test_l


# 神经网络定义
class CNN(nn.Module):
    def __init__(self, num_classes=5):  # 假设有5个类别
        super(CNN, self).__init__()
        # 第一层卷积层，输入通道3（RGB图像），输出通道32，卷积核大小3x3，填充1
        self.con1 = nn.Conv2d(3, 32, 3, padding=1)
        # 第二层卷积层，输入通道32，输出通道64，卷积核大小3x3，填充1
        self.con2 = nn.Conv2d(32, 64, 3, padding=1)
        # 第三层卷积层，输入通道64，输出通道128，卷积核大小3x3，填充1
        self.con3 = nn.Conv2d(64, 128, 3, padding=1)
        # 最大池化层，池化窗口大小2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层，输入大小128*28*28，输出512
        self.connect1 = nn.Linear(128 * 28 * 28, 512)  # 输入层维度根据实际情况调整
        # 全连接层，输入512，输出类别数
        self.connect2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 前向传播：卷积层1 -> ReLU -> 池化层
        x = self.pool(F.relu(self.con1(x)))
        # 前向传播：卷积层2 -> ReLU -> 池化层
        x = self.pool(F.relu(self.con2(x)))
        # 前向传播：卷积层3 -> ReLU -> 池化层
        x = self.pool(F.relu(self.con3(x)))
        # 展平特征图以输入全连接层
        x = x.view(-1, 128 * 28 * 28)  # Flatten layer
        # 前向传播：全连接层1 -> ReLU
        x = F.relu(self.connect1(x))
        # 前向传播：全连接层2（输出层）
        x = self.connect2(x)
        return x


# 训练模型
def ModelTrain(model, train_l, criterion, optim, num_epochs=20):
    model.train()
    losses = []  # 记录每个epoch的损失
    accuracies = []  # 记录每个epoch的准确率
    for e in range(num_epochs):
        running_loss = 0.0
        correct = total = 0
        for i, l in train_l:
            # 将数据加载到设备（GPU或CPU）
            i, l = i.to(device), l.to(device)
            optim.zero_grad()  # 梯度清零
            outputs = model(i)  # 前向传播
            loss = criterion(outputs, l)  # 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            running_loss += loss.item()

            _, predi = torch.max(outputs, 1)  # 获取预测标签
            total += l.size(0)
            correct += (predi == l).sum().item()  # 统计正确预测的数量

        loss_e = running_loss / len(train_l)  # 计算每个epoch的平均损失
        accuracy_e = 100 * correct / total  # 计算每个epoch的准确率
        losses.append(loss_e)
        accuracies.append(accuracy_e)
        print(f'Epoch {e + 1}/{num_epochs}, Loss: {loss_e:.4f}, Accuracy: {accuracy_e:.2f}%')

    return losses, accuracies


# 评估模型
def evaluate_model(model, test_l):
    model.eval()
    total = correct = 0
    with torch.no_grad():  # 评估时不需要计算梯度
        for imgs, labels in test_l:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predi = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predi == labels).sum().item()
    accuracy = 100 * correct / total  # 计算测试集的准确率
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy


# 主函数
if __name__ == '__main__':
    data_dir = './data'
    # 检查是否可以使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型并将其加载到设备
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
    optim = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    # 加载数据
    train_l, test_l = load_data(data_dir)
    # 训练模型
    losses, accuracies = ModelTrain(model, train_l, criterion, optim)
    # 评估模型
    accuracy = evaluate_model(model, test_l)

    # 绘制损失曲线和准确率曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')

    plt.show()