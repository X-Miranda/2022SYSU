import numpy as np
import matplotlib.pyplot as plt
import time
from math import *


def f0(A, b, x, lamda):
    """
    计算目标函数值：
    f(x) = 0.5 * sum( ||A[i] @ x - b[i]||_2^2 ) + lamda * ||x||_1
    """
    residuals = A @ x - b  # 计算残差矩阵，形状为 (n, d1)
    data_term = 0.5 * np.sum(residuals ** 2)  # 计算平方误差的总和
    regularization_term = lamda * np.linalg.norm(x, ord=1)  # 计算L1正则化项
    return data_term + regularization_term


def soft_thresholding(x, threshold):
    """
    软门限函数
    :param x: 输入向量
    :param threshold: 阈值
    :return: 软门限处理后的向量
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def proximal_gradient_method(A, b, lamda, alpha=0.0001, max_iter=5000, tol=1e-5, if_draw=True):
    """
    邻近点梯度法求解
    :param A: 矩阵，形状为 (n, d1, d2)
    :param b: 向量，形状为 (n, d1)
    :param lamda: 正则化参数
    :param alpha: 学习率
    :param max_iter: 最大迭代次数
    :param tol: 收敛阈值
    :return: 最优解 x, 每次迭代的解与真实解的距离, 每次迭代的解与最优解的距离
    """
    start_time = time.time()

    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解
    iterates = []  # 记录每步的解

    # 迭代求解
    for _ in range(max_iter):
        x_old = x.copy()

        # 计算梯度
        gradient = np.sum([A[i].T @ (A[i] @ x - b[i]) for i in range(n)], axis=0)

        # 软门限法算argmin
        x = soft_thresholding(x - alpha * gradient, lamda * alpha)

        iterates.append(x)  # 记录解

        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break

    end_time = time.time()
    diff_time = end_time - start_time

    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]

    if if_draw:
        # 绘制距离图
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('Proximal Gradient Method')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()

    print(f'Proximal Gradient Time(lambda={lamda}): {diff_time} s')  # 打印时间
    print(f'Distance(lambda={lamda}): '
          f'{np.linalg.norm(x - x_true)}', end='\n\n')  # 打印二范数误差

    return x, distances_to_true, distances_to_opt



def admm(A, b, lamda, C=1, max_iter=1000, tol=1e-5, if_draw=True):
    """
    交替方向乘子法（ADMM）求解
    :param A: 矩阵，形状为 (n, d1, d2)
    :param b: 向量，形状为 (n, d1)
    :param lamda: 正则化参数
    :param C: 惩罚参数
    :param max_iter: 最大迭代次数
    :param tol: 收敛阈值
    :return: 最优解 x, 每次迭代的解与真实解的距离, 每次迭代的解与最优解的距离
    """
    start_time = time.time()

    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解
    y = np.zeros(d2)  # 辅助变量
    v = np.zeros(d2)  # 对偶变量

    iterates = []  # 记录每步的解
    r = []  # 记录目标函数值

    # 预计算矩阵和向量
    A_T_A = np.sum([A[i].T @ A[i] for i in range(n)], axis=0)
    A_T_b = np.sum([A[i].T @ b[i] for i in range(n)], axis=0)
    inv_matrix = np.linalg.inv(A_T_A + C * np.eye(d2))

    for _ in range(max_iter):
        x_old = x.copy()

        # 更新 x
        x = inv_matrix @ (A_T_b + C * y - v)

        # 更新 y
        y = soft_thresholding(x + v / C, lamda / C)

        # 更新 v
        v += C * (x - y)

        iterates.append(x)
        r.append(f0(A, b, x, lamda))

        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break

    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]

    end_time = time.time()
    diff_time = end_time - start_time

    if if_draw:
        # 绘制距离变化图
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('ADMM')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()

    print(f"ADMM Time (lambda={lamda}): {diff_time} s")  # 打印所用时间
    print(f"Distance (lambda={lamda}): {np.linalg.norm(x - x_true)}\n")  # 打印二范数误差

    return x, distances_to_true, distances_to_opt



def subgradient(A, b, lamda, alpha=0.0001, max_iter=5000, tol=1e-5, if_draw=True):
    """
    次梯度法求解
    :param A: 矩阵，形状为 (n, d1, d2)
    :param b: 向量，形状为 (n, d1)
    :param lamda: 正则化参数
    :param alpha: 学习率
    :param max_iter: 最大迭代次数
    :param tol: 收敛阈值
    :return: 最优解 x, 每次迭代的解与真实解的距离, 每次迭代的解与最优解的距离
    """
    start_time = time.time()

    n, d1, d2 = A.shape
    x = np.zeros(d2)  # 初始解
    iterates = []  # 记录每步的解

    # 预计算梯度相关的矩阵和向量
    A_T_A = np.sum([A[i].T @ A[i] for i in range(n)], axis=0)
    A_T_b = np.sum([A[i].T @ b[i] for i in range(n)], axis=0)

    for _ in range(max_iter):
        x_old = x.copy()

        # 计算次梯度
        gradient = A_T_A @ x - A_T_b
        subgrad = np.sign(x)
        subgrad[x == 0] = np.random.uniform(-1, 1, size=np.sum(x == 0))  # 对于 x=0 的情况，随机选择 [-1, 1] 之间的值
        g = gradient + lamda * subgrad

        # 更新 x
        x = x - alpha * g

        iterates.append(x)

        # 判断收敛
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break

    end_time = time.time()
    diff_time = end_time - start_time

    # 计算每步解与真实解之间以及最优解之间的距离
    distances_to_true = [np.linalg.norm(iterate - x_true, ord=2) for iterate in iterates]
    distances_to_opt = [np.linalg.norm(iterate - x, ord=2) for iterate in iterates]

    if if_draw:
        # 绘制距离变化图
        plt.figure()
        plt.plot(distances_to_true, label='distance to true')
        plt.plot(distances_to_opt, label='distance to optimal')
        plt.title('Subgradient')
        plt.xlabel('iteration')
        plt.ylabel('distance')
        plt.grid()
        plt.legend()
        plt.show()

    print(f"Subgradient Time (lambda={lamda}): {diff_time} s")  # 打印时间
    print(f"Distance (lambda={lamda}): {np.linalg.norm(x - x_true)}\n")  # 打印二范数误差

    return x, distances_to_true, distances_to_opt



def adjust_lamda(A, b, lamdas, method):
    '''
    调整正则化参数，lamdas是参数列表，method决定用哪个优化算法，同时作为绘制图形的suptitle，
    method只能取值'proximal gradient', 'admm' 或'subgradient'
    '''
    fig, axes = plt.subplots(int(sqrt(len(lamdas))), ceil(len(lamdas) / 2), figsize=(12, 8))  # 创建多个子图
    # 画每一个参数值对应的子图
    for i, lamda in enumerate(lamdas):
        if method == 'Proximal Gradient':
            r1 = proximal_gradient_method(A, b, lamda, if_draw=False)
        elif method == 'ADMM':
            r1 = admm(A, b, lamda, if_draw=False)
        elif method == 'Subgradient':
            r1 = subgradient(A, b, lamda, if_draw=False)

        row, col = divmod(i, ceil(len(lamdas) / 2))  # 计算子图位置
        axes[row, col].plot(r1[1], label='distance to true')
        axes[row, col].plot(r1[2], label='distance to opt')
        axes[row, col].set_title(r"$\lambda = $" + f"{lamda}")
        axes[row, col].set_xlabel('iteration')
        axes[row, col].set_ylabel('distance')
        axes[row, col].grid()
        axes[row, col].legend()

    plt.suptitle(method)
    plt.tight_layout()
    plt.show()



np.random.seed(0)
num = 10
d1 = 5
d2 = 200
lamda = 1  # 可以调整

# 随机生成矩阵A、x的真值、和向量b
A = np.array([np.random.normal(0, 1, (d1, d2)) for _ in range(num)])
x_true = np.zeros(d2)
# 随机选择5个位置非0，其他位置为0
nonzero_indices = np.random.choice(d2, 5, replace=False)
x_true[nonzero_indices] = np.random.normal(0, 1, d1)
b = np.array([A[i].dot(x_true) + np.random.normal(0, 0.1, d1) for i in range(num)])

# 三种算法求解
x_opt1 = proximal_gradient_method(A, b, lamda)[0]
x_opt2 = admm(A, b, lamda)[0]
x_opt3 = subgradient(A, b, lamda)[0]

# 调整正则化参数
#lamdas = [0.001, 0.01, 0.1, 1, 10, 100]
#adjust_lamda(A, b, lamdas, 'Subgradient')