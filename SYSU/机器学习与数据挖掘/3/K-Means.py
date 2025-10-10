import numpy as np
import pandas as pd
import time

def initialize_centroids_plusplus(X, k):
    """使用K-Means++初始化方法"""
    centroids = [X[np.random.randint(X.shape[0])]]
    for _ in range(1, k):
        distances = np.min(np.sum((X - np.array(centroids)[:, np.newaxis])**2, axis=2), axis=0)
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        index = np.searchsorted(cumulative_probabilities, r)
        centroids.append(X[index])
    return np.array(centroids)


def k_means(X, k, max_iter=100, init_method='random'):
    """K-Means算法"""
    start_time = time.time()  # 记录开始时间
    if init_method == 'random':
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    elif init_method == 'k-means++':
        centroids = initialize_centroids_plusplus(X, k)

    for _ in range(max_iter):
        # 分配阶段
        clusters = np.array([np.argmin([np.inner(c - x, c - x) for c in centroids]) for x in X])
        # 更新阶段
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        # 检查收敛
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间

    return clusters, centroids, elapsed_time

# 加载数据
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# 分离特征和标签
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# 数据归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

# 使用K-Means++初始化方法进行K-Means聚类
clusters_kmeanspp, centroids_kmeanspp, time_kmeanspp = k_means(X_train, 10, init_method='k-means++')

# 使用随机初始化方法进行K-Means聚类
clusters_random, centroids_random, time_random = k_means(X_train, 10, init_method='random')

def calculate_accuracy(y_true, clusters):
    """计算聚类精度"""
    y_pred = np.zeros_like(y_true)
    for i in range(10):
        mask = (clusters == i)
        y_pred[mask] = np.bincount(y_true[mask]).argmax()
    return np.mean(y_pred == y_true)

# 计算聚类精度
acc_kmeanspp = calculate_accuracy(y_train, clusters_kmeanspp)
acc_random = calculate_accuracy(y_train, clusters_random)

# 打印结果
print(f'K-Means++聚类精度: {acc_kmeanspp:.4f}, 运行时间: {time_kmeanspp:.4f}秒')
print(f'随机初始化聚类精度: {acc_random:.4f}, 运行时间: {time_random:.4f}秒')
