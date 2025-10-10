import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import time

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

def calculate_accuracy(y_true, clusters):
    """计算聚类精度"""
    y_pred = np.zeros_like(y_true)
    for i in range(10):
        mask = (clusters == i)
        y_pred[mask] = np.bincount(y_true[mask]).argmax()
    return np.mean(y_pred == y_true)


def initialize_gmm_params(X, k, cov_type='diagonal_equal'):
    """初始化GMM参数"""
    n_features = X.shape[1]
    weights = np.ones(k) / k
    means = X[np.random.choice(X.shape[0], k, replace=False)]
    covariances = []

    for i in range(k):
        if cov_type == 'diagonal_equal':
            # 对角且元素值都相等
            cov = np.eye(n_features) * np.var(X)
        elif cov_type == 'diagonal':
            # 对角但对元素值不要求相等
            cov = np.diag(np.var(X, axis=0))
        elif cov_type == 'full':
            # 普通矩阵
            diff = X - means[i]
            cov = np.dot(diff.T, diff) / (X.shape[0] - 1)
        else:
            raise ValueError("Invalid covariance matrix type")

        # 添加正则化确保协方差矩阵是正定的
        cov = cov + 1e-6 * np.eye(n_features)
        covariances.append(cov)

    return weights, means, covariances


def e_step(X, weights, means, covariances):
    """E步骤"""
    n_samples = X.shape[0]
    k = len(weights)
    responsibilities = np.zeros((n_samples, k))
    for i in range(k):
        try:
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
        except np.linalg.LinAlgError:
            responsibilities[:, i] = 0  # 如果协方差矩阵是奇异的，设置责任度为0
    responsibilities /= (responsibilities.sum(axis=1, keepdims=True) + 1e-10)  # 添加小的正数避免除以零
    return responsibilities

def m_step(X, responsibilities):
    """M步骤"""
    k = responsibilities.shape[1]
    n_features = X.shape[1]
    weights = responsibilities.mean(axis=0)
    means = np.dot(responsibilities.T, X) / (responsibilities.sum(axis=0)[:, np.newaxis] + 1e-10)
    covariances = []
    for i in range(k):
        diff = X - means[i]
        cov = np.dot(responsibilities[:, i] * diff.T, diff) / (responsibilities[:, i].sum() + 1e-10)
        # 添加正则化确保协方差矩阵是正定的
        cov = cov + 1e-6 * np.eye(n_features)
        covariances.append(cov)
    return weights, means, covariances

def gmm(X, k, cov_type='diagonal_equal', max_iter=100):
    """GMM算法"""
    start_time = time.time()  # 记录开始时间
    weights, means, covariances = initialize_gmm_params(X, k, cov_type)
    for iteration in range(max_iter):
        # E步骤
        responsibilities = e_step(X, weights, means, covariances)
        # M步骤
        weights, means, covariances = m_step(X, responsibilities)
        # 检查是否所有责任度都有效
        if np.any(np.isnan(responsibilities)) or np.any(np.isinf(responsibilities)):
            break

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间

    return responsibilities, means, elapsed_time

def run_gmm_experiments(X, y, k, n_experiments=10):
    for cov in {'full', 'diagonal_equal', 'diagonal'}:
        accuracies = []
        times = []
        for seed in range(n_experiments):
            np.random.seed(seed)
            responsibilities, gmm_means, time_gmm = gmm(X, k, cov_type=cov)
            gmm_clusters = np.argmax(responsibilities, axis=1)
            acc_gmm = calculate_accuracy(y, gmm_clusters)
            accuracies.append(acc_gmm)
            times.append(time_gmm)
        print(f'GMM_{cov}聚类精度: {np.mean(accuracies):.4f}, 运行时间: {np.mean(times):.4f}秒')

run_gmm_experiments(X_train, y_train, 10)


