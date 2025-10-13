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

# 逻辑回归类
class logic:
    def __init__(self, input_dim, lv=0.1):
        np.random.seed(0) # 设置随机数种子，保证结果可复现
        self.weights = np.random.randn(input_dim, 1) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((1, 1))
        self.lv = lv

    # Sigmoid激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 前向传播过程
    def forward(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)

    # 计算损失
    @staticmethod
    def loss(y_pre, y):
        return -np.mean(y * np.log(y_pre) + (1 - y) * np.log(1 - y_pre))

    # 反向传播，更新参数
    def back(self, x, y):
        y_pre = self.forward(x)  # 计算预测值
        dz = y_pre - y  # 计算梯度
        d_w = np.dot(x.T, dz) / x.shape[0]  # 计算权重的梯度
        d_bias = np.sum(dz) / x.shape[0]  # 计算偏置的梯度
        # 更新参数
        self.weights -= self.lv * d_w
        self.bias -= self.lv * d_bias

    # 训练模型
    def train(self, X, Y, epochs, size):
        losses = []
        best_loss = float('inf')
        best_e = 0
        best_w = None
        best_bias = None

        for epoch in range(epochs):
            cur_loss = 0
            for i in range(0, X.shape[0], size):
                x_batch = X.iloc[i : i + size].values
                y_batch = Y.iloc[i : i + size].values.reshape(-1, 1)

                y_pre = self.forward(x_batch)
                # 计算损失
                cur_loss = self.loss(y_pre, y_batch)
                # 反向传播
                self.back(x_batch, y_batch)

            losses.append(cur_loss)
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_e = epoch + 1
                best_w = self.weights.copy()
                best_bias = self.bias.copy()

            print(f"Epoch: {epoch + 1}, Loss: {cur_loss}")

        # 使用最佳参数
        self.weights = best_w
        self.bias = best_bias

        return losses, best_e, best_loss

    # 预测结果
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(np.int32)

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号



# 模型训练
input_dim = X.shape[1]  # 获取特征的维度
model = logic(input_dim)  # 创建逻辑回归模型实例
epochs = 3000  # 设置训练的迭代次数
size = 64  # 设置每次迭代的批量大小

losses, best_e, best_loss = model.train(X, Y, epochs, size)

# 进行预测并计算准确率
results = model.predict(X)
accuracy = (results == np.array(Y).reshape(-1, 1)).astype(np.int32).mean()
print(f"最终准确率：{accuracy * 100:.2f}%，最佳loss：{best_loss}")

# 提取权重和偏置用于绘制决策边界
w = model.weights.flatten()
b = model.bias.flatten()

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, label="训练损失")
plt.axvline(best_e, color="green", linestyle="--", label=f"最佳Epoch: {best_e}")
plt.xlabel("迭代次数")
plt.ylabel("损失值")
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
ax2.scatter(
    feature1[results == 1], feature2[results == 1], color="blue", label="已购买"
)
ax2.set_title("预测结果")
ax2.set_xlabel("年龄")
ax2.set_ylabel("收入")
ax2.legend()

# 绘制决策边界
x_values = np.array([X.iloc[:, 0].min(), X.iloc[:, 0].max()])
y_values = -(w[0] / w[1]) * x_values - (b / w[1])
y_values = y_values.flatten()  # 确保 y_values 是一维的

ax2.plot(x_values, y_values, color='red', label='决策边界')
ax2.legend()

plt.show()
