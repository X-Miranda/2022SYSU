import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# 加载数据
train_data = pd.read_csv('mnist_01_train.csv')
test_data = pd.read_csv('mnist_01_test.csv')

# 提取特征和标签
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用线性核函数的SVM
svm_linear = SVC(kernel='linear', C=4.0)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print(f"线性核函数SVM的准确率: {accuracy_score(y_test, y_pred_linear)}")

# 使用高斯核函数的SVM
svm_rbf = SVC(kernel='rbf', C=4.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(f"高斯核函数SVM的准确率: {accuracy_score(y_test, y_pred_rbf)}")


# Hinge loss的线性分类器实现
class HingeLossClassifier:
    def __init__(self, learning_rate=0.0001, n_iters=360):  # 调整学习率和迭代次数
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.W = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                if y[idx] * np.dot(x_i, self.W) < 1:
                    self.W -= self.learning_rate * (2 * self.W - np.dot(y[idx], x_i))
                else:
                    self.W -= self.learning_rate * 2 * self.W

    def predict(self, X):
        linear_output = np.dot(X, self.W)
        return np.where(linear_output >= 0, 1, 0)

class CrossEntropyLossClassifier:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # 将标签转换为-1和1

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                loss = -y_[idx] * np.log(1 / (1 + np.exp(-linear_output))) if y_[idx] == 1 else -(1 - y_[idx]) * np.log(1 / (1 + np.exp(-linear_output)))
                dw = -self.learning_rate * (y_[idx] * x_i) if loss > 0 else 0
                db = -self.learning_rate * y_[idx] if loss > 0 else 0


                self.weights -= dw
                self.bias -= db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(1 / (1 + np.exp(-linear_output)))


lc_hinge = HingeLossClassifier()
lc_hinge.fit(X_train, y_train)
y_pred_hinge = lc_hinge.predict(X_test)
accuracy_hinge = accuracy_score(y_test, y_pred_hinge)

# 使用cross-entropy loss训练
lc_cross_entropy = CrossEntropyLossClassifier()
lc_cross_entropy.fit(X_train, y_train)
y_pred_cross_entropy = lc_cross_entropy.predict(X_test)
accuracy_cross_entropy = accuracy_score(y_test, y_pred_cross_entropy)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


#评估模型性能
def evaluate_model(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return conf_matrix, precision, recall, f1

# 使用 hinge loss 的评估结果
conf_matrix_hinge, precision_hinge, recall_hinge, f1_hinge = evaluate_model(y_test, y_pred_hinge)
print(f"hinge loss - 混淆矩阵:\n{conf_matrix_hinge}")
print(f"hinge loss - 精确率: {precision_hinge}")
print(f"hinge loss - 召回率: {recall_hinge} ")
print(f"hinge loss - F1分数: {f1_hinge}")

# 使用 cross-entropy loss 的评估结果
conf_matrix_cross_entropy, precision_cross_entropy, recall_cross_entropy, f1_cross_entropy = evaluate_model(y_test, y_pred_cross_entropy)
print(f"cross-entropy loss - 混淆矩阵:\n{conf_matrix_cross_entropy}")
print(f"cross-entropy loss - 精确率: {precision_cross_entropy} ")
print(f"cross-entropy loss - 召回率: {recall_cross_entropy}")
print(f"cross-entropy loss - F1分数: {f1_cross_entropy}")

# 比较两种损失函数的 F1 分数
if f1_hinge > f1_cross_entropy:
    print("Hinge loss better")
elif f1_hinge < f1_cross_entropy:
    print("Cross-Entropy better")
else:
    print("Same")

print(f"hinge loss 准确率: {accuracy_hinge}")
print(f"cross-entropy loss 准确率: {accuracy_cross_entropy}")