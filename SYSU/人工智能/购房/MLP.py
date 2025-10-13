import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 预处理各项数据
data = pd.read_csv("data.csv")
X = data.iloc[:, :-1]  # 去除最后一列作为特征
Y = data.iloc[:, -1]  # 最后一列作为标签
X_min = X.min()  # 计算特征最小值
X_max = X.max()  # 计算特征最大值
X = (X - X_min) / (X_max - X_min)  # 特征归一化

# 模型类
class MLP:
    def __init__(self, layers, lv=0.1):
        np.random.seed(0)
        self.layers = layers # 层数
        self.lv = lv  # 学习率
        self.weights = []  # 权重
        self.bias = []  # 偏置
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2. / layers[i])  # 初始化权重
            self.weights.append(weight)
            bias = np.zeros((1, layers[i + 1]))  # 初始化偏置
            self.bias.append(bias)

    # Sigmoid激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # ReLu激活函数
    def relu(self, z):
        return np.maximum(0, z)

    # 前向传播过程
    def forward(self, x):
        y = x
        self.cache = {'A0': x}  # 缓存每一层的输出结果
        for i in range(len(self.weights) - 1):
            z = np.dot(y, self.weights[i]) + self.bias[i]
            y = self.relu(z)
            self.cache[f'Z{i + 1}'] = z
            self.cache[f'A{i + 1}'] = y
        z = np.dot(y, self.weights[-1]) + self.bias[-1]
        y = self.sigmoid(z)
        self.cache[f'Z{len(self.weights)}'] = z
        self.cache[f'A{len(self.weights)}'] = y
        return y

    # 计算损失
    def loss(self, y_pred, y):
        return np.mean((y_pred - y) ** 2)

    # 反向传播，更新参数
    def back(self, x, y):
        m = x.shape[0]  # 样本数
        output = self.cache[f'A{len(self.weights)}']  # 输出层输出
        errors = [None] * len(self.weights)

        errors[-1] = (output - y) * (output * (1 - output))  # 计算输出层的误差

        for i in reversed(range(len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * (self.cache[f'A{i + 1}'] > 0)  # 计算隐藏层的误差

        for i in range(len(self.weights)):
            grad_weight = np.dot(self.cache[f'A{i}'].T, errors[i]) / m  # 计算权重梯度
            grad_bias = np.sum(errors[i], axis=0, keepdims=True) / m  # 计算偏置梯度
            self.weights[i] -= self.lv * grad_weight  # 更新权重
            self.bias[i] -= self.lv * grad_bias  # 更新偏置

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 模型训练
layers = [2, 16, 1]  # 输入层，一个隐藏层，一个输出层
model = MLP(layers)  # 初始化模型
epochs = 2000
size = 64
losses = []  # 记录损失值
best_loss = float('inf')  # 初始最佳损失
best_e = 0  # 最佳Epoch
best_w = []  # 保存每层的最佳权重
best_bias = []  # 保存每层的最佳偏置

for epoch in range(epochs):
    for i in range(0, X.shape[0], size):
        x_batch = X.iloc[i: i + size].values
        y_batch = Y.iloc[i: i + size].values.reshape(-1, 1)

        y_pre = model.forward(x_batch)  # 前向传播
        cur_loss = model.loss(y_pre, y_batch)  # 计算当前损失
        model.back(x_batch, y_batch)  # 反向传播更新参数

    losses.append(cur_loss)
    if cur_loss < best_loss:
        best_loss = cur_loss
        best_e = epoch + 1
        best_w = [w.copy() for w in model.weights]  # 深复制每层权重
        best_bias = [b.copy() for b in model.bias]  # 深复制每层偏置

# 使用最佳参数进行预测
model.weights = best_w
model.bias = best_bias

results = (model.forward(X) > 0.5).astype(np.int32)
accuracy = (results == np.array(Y).reshape(-1, 1)).astype(np.int32).mean()
print(f"最终准确率：{accuracy * 100:.2f}%，最佳loss：{best_loss}")

# 提取第一层的权重和偏置，因为输入层直接连接输出层
w = model.weights[0].flatten()
b = model.bias[0].flatten()

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, label="训练损失")
plt.axvline(best_e, color="green", linestyle="--", label=f"最佳Epoch: {best_e}")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.text(epochs, losses[-1], f"最终损失: {losses[-1]:.4f}", ha="right", va="baseline")
plt.legend()

results = results.flatten()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

feature1 = X.iloc[:, 0]
feature2 = X.iloc[:, 1]

ax1.scatter(feature1[Y == 0], feature2[Y == 0], color="red", label="未购买")
ax1.scatter(feature1[Y == 1], feature2[Y == 1], color="blue", label="已购买")
ax1.set_title("真实情况")
ax1.set_xlabel("年龄")
ax1.set_ylabel("收入")
ax1.legend()

ax2.scatter(feature1[results == 0], feature2[results == 0], color="red", label="未购买")
ax2.scatter(feature1[results == 1], feature2[results == 1], color="blue", label="已购买")
ax2.set_title("预测结果")
ax2.set_xlabel("年龄")
ax2.set_ylabel("收入")
ax2.legend()

plt.show()

