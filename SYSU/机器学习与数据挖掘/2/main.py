import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle


def load_data(dir):
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_train.append(dict[b'data'])
        Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)

    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X_test = dict[b'data']
    Y_test = dict[b'labels']

    return X_train, Y_train, X_test, Y_test

# 读取数据
X_train, Y_train, X_test, Y_test = load_data('./data')

# 数据预处理
X_train = X_train.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
X_test = X_test.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
Y_train = np.eye(10)[Y_train]
Y_test = np.eye(10)[Y_test]

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train)
Y_train_tensor = torch.tensor(Y_train)
X_test_tensor = torch.tensor(X_test)
Y_test_tensor = torch.tensor(Y_test)

# 创建数据集和 DataLoader
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc(x)
        return x


# 修改后的MLP类
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加批量归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # 添加Dropout
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # 确保输入特征维度正确
        return self.network(x)


class ModifiedLeNet(nn.Module):
    def __init__(self, num_conv_layers=2):
        super(ModifiedLeNet, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.layers = nn.ModuleList()
        in_channels = 3
        out_channels = 32

        for i in range(num_conv_layers):
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 120)  # Adjust the input size based on the number of conv layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(-1, 256 * 4 * 4)  # Adjust the size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, optimizer, criterion, num_epochs=10):

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)  # 得到预测结果
            total += target.size(0)
            correct += (predicted == target.argmax(dim=1)).sum().item()  # 修正比较逻辑

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the 10000 test images: {accuracy} %")
    return accuracy


# 训练和评估 Softmax 分类器
model = SoftmaxClassifier()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_model(model, optimizer, criterion)
evaluate_model(model, train_loader)
evaluate_model(model, test_loader)

'''
# 训练和评估 MLP
input_size = 3 * 32 * 32  # 输入特征维度
hidden_sizes = [1024, 512, 256]  # 隐藏层尺寸
output_size = 10  # 输出类别数量

model = MLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 尝试使用Adam优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 添加学习率调度器
num_epochs = 20

train_model(model, optimizer, criterion)
print("Accuracy of the trained model is:")
evaluate_model(model, train_loader)
print("Accuracy of the test model is:")
evaluate_model(model, test_loader)

# 训练和评估 CNN
model = ModifiedLeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
num_epochs = 10  # 增加训练轮数

train_model(model, optimizer, criterion, num_epochs)
print("Accuracy of the trained model is:")
evaluate_model(model, train_loader)
print("Accuracy of the test model is:")
evaluate_model(model, test_loader)'''
