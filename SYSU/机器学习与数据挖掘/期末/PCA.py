# coding:utf-8
import os
from numpy import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


def img2vector(image):
    """
    将图像文件转换为一维向量。
    参数:image (str): 图像文件的路径。
    返回:imgVector (numpy.ndarray): 一维向量形式的图像数据。
    """
    # 使用 OpenCV 读取图像，0 表示以灰度模式读取
    img = cv2.imread(image, 0)

    # 将图像矩阵直接展平为一维向量
    imgVector = img.flatten().reshape(1, -1)  # 使用 flatten 展平，然后重新塑形为 (1, rows*cols)

    return imgVector

# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_orl(k):
    """
    从 ORL 数据集中加载人脸图像，并将每个人随机选择的 k 张图像作为训练集，其余作为测试集。

    参数:
        k (int): 每个人选择的训练样本数量。
    返回:
        train_face (numpy.ndarray): 训练集图像数据。
        train_label (numpy.ndarray): 训练集标签。
        test_face (numpy.ndarray): 测试集图像数据。
        test_label (numpy.ndarray): 测试集标签。
    """
    orlpath = "./orl_faces"  # 数据集路径
    num_people = 40  # 总人数
    num_images = 10  # 每个人的图像数量
    image_size = 112 * 92  # 图像尺寸

    # 初始化训练集和测试集
    train_face = np.zeros((num_people * k, image_size))
    train_label = np.zeros(num_people * k, dtype=int)
    test_face = np.zeros((num_people * (num_images - k), image_size))
    test_label = np.zeros(num_people * (num_images - k), dtype=int)

    for i in range(num_people):  # 遍历每个人
        people_num = i + 1
        image_indices = list(range(1, num_images + 1))  # 图像编号从1到10
        random.shuffle(image_indices)  # 随机打乱图像编号

        # 分配训练集和测试集
        train_indices = image_indices[:k]  # 前 k 张作为训练集
        test_indices = image_indices[k:]  # 剩余的作为测试集

        for j, idx in enumerate(train_indices):  # 遍历训练集图像
            image_path = os.path.join(orlpath, f"s{people_num}", f"{idx}.pgm")
            img_vector = img2vector(image_path)  # 将图像转换为一维向量
            train_face[i * k + j] = img_vector
            train_label[i * k + j] = people_num

        for j, idx in enumerate(test_indices):  # 遍历测试集图像
            image_path = os.path.join(orlpath, f"s{people_num}", f"{idx}.pgm")
            img_vector = img2vector(image_path)  # 将图像转换为一维向量
            test_face[i * (num_images - k) + j] = img_vector
            test_label[i * (num_images - k) + j] = people_num

    return train_face, train_label, test_face, test_label


def PCA(data, r):
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)

    # 数据标准化
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 去中心化

    C = A * A.T  # 计算协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    V_r = V[:, 0:r]  # 取前r个特征向量
    V_r = A.T * V_r  # 将小矩阵特征向量转换为大矩阵特征向量

    # 归一化特征向量
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])

        # 计算降维后的数据
    final_data = A * V_r

    return final_data, data_mean, V_r

def plot_accuracy(x_value, y_value, r):
    plt.plot(x_value, y_value, marker="o", markerfacecolor="red")
    for a, b in zip(x_value, y_value):
        plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    plt.title(f"降到{r}维时识别准确率", fontsize=14)
    plt.xlabel("k值", fontsize=14)
    plt.ylabel("准确率", fontsize=14)
    plt.show()

# 人脸识别
def face_rec():
    dims = range(10, 41, 10)  # 降维维度
    accuracy_list = []
    x_value_list = []

    for r in dims:
        print(f"\n降到{r}维时")
        x_value = []
        y_value = []
        for k in range(1, 10):
            train_face, train_label, test_face, test_label = load_orl(k)  # 得到数据集

            # 利用PCA算法进行训练
            data_train_new, data_mean, V_r = PCA(train_face, r)
            num_train = data_train_new.shape[0]  # 训练脸总数
            num_test = test_face.shape[0]  # 测试脸总数
            temp_face = test_face - np.tile(data_mean, (num_test, 1))
            data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
            data_test_new = np.array(data_test_new)  # mat change to array
            data_train_new = np.array(data_train_new)

            # 测试准确度
            true_num = 0
            for i in range(num_test):
                testFace = data_test_new[i, :]
                diffMat = data_train_new - np.tile(testFace, (num_train, 1))  # 训练数据与测试脸之间距离
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
                sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
                indexMin = sortedDistIndicies[0]  # 距离最近的索引
                if train_label[indexMin] == test_label[i]:
                    true_num += 1

            accuracy = true_num / num_test
            x_value.append(k)
            y_value.append(round(accuracy, 2))

            print(f'每人选择{k}张照片进行训练，准确率为: {accuracy * 100:.4f}%')

        # 绘图
        plot_accuracy(x_value, y_value, r)
        accuracy_list.append(y_value)
        x_value_list = x_value  # 所有维度的x值是一样的

    # 各维度下准确度比较
    lines = []
    labels = []
    for i, r in enumerate(dims):
        line, = plt.plot(x_value_list, accuracy_list[i], marker="o", markerfacecolor="pink")
        lines.append(line)
        labels.append(f"降到{r}维")

    plt.legend(lines, labels, loc=4)
    plt.title("各维度识别准确率比较", fontsize=14)
    plt.xlabel("k值", fontsize=14)
    plt.ylabel("准确率", fontsize=14)
    plt.show()

if __name__ == '__main__':
    face_rec()