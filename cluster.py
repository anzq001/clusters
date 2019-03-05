# 聚类算法
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kmeans import my_kmeans
from dbscan import my_dbscan
DISTANCE = 5
marker = ["o", "v", "*", "D", "d", ".", ",", "^"]
color = ["r", "g", "b", "y", "m", "black", "orange", "gold"]
FIGURE_INDEX = 0
class_num = 2
tags = list(range(class_num))


def generate_dots1(dots_num, class_num):
    """
    产生dots_nums个点，分属于class_num个类，其中默认每个类之间的x平均距离为DISTANCE，y平均距离为0
    :param dots_num:
    :param class_num:
    :return: 二维数组，其中每一行表示一个点
    """
    centers = [i * DISTANCE for i in range(class_num)]
    dots = []
    tags = []
    for index in range(dots_num):
        tag = index % class_num
        center = centers[tag]
        tags.append(tag)
        dots.append([center + random.gauss(0, 1), random.gauss(0, 1)])
    random.shuffle(list(zip(dots, tags)))
    return dots, tags

def generate_dots2(dots_num):
    """
    产生分布为环状的点，用于dbscan算法的训练
    :param dots_num: 产生点的数量
    :return: 产生的二维点构成的数组
    """
    radiuses = random.choices([5, 10], k=dots_num)
    tags = [int(ele == 5) for ele in radiuses]
    rads = random.choices(np.arange(0, 1, 0.01) * 2 * np.pi, k=dots_num)
    radiuses = radiuses + np.random.randn(dots_num) * 0.5
    rads = rads + np.random.randn(dots_num) * 0.01
    dots = radiuses * np.array([np.sin(rads), np.cos(rads)])
    return dots.T, tags


def scatter_dots(dots, tags, new_plot=True, title=None, subplot=None):
    if new_plot:
        global FIGURE_INDEX
        FIGURE_INDEX += 1
    plt.figure(FIGURE_INDEX)
    if subplot is not None:
        plt.subplot(subplot)
    for dot, tag in zip(dots, tags):
        plt.scatter(dot[0], dot[1], marker=marker[tag], c=color[tag])


if __name__ == "__main__":
    # 生成100个点
    dots, tags = generate_dots1(dots_num=200, class_num=class_num)
    scatter_dots(dots, tags, new_plot=True, subplot=221, title="原始点集")

    #利用sklearn算法中的KMeans类对dots做分类
    model = KMeans(class_num)
    pred_tag = model.fit_predict(dots)
    scatter_dots(dots, pred_tag, new_plot=False, subplot=222, title="sklearn库中kmeans分类后的点集")

    #利用自己建立的kmeans模型对dots做分类
    pred_tags = my_kmeans(dots, class_num)
    scatter_dots(dots, pred_tags, new_plot=False, subplot=223, title="自己训练得到的kmeans分类后的点集")
    plt.show()

    dots1, tags1 = generate_dots2(200)
    scatter_dots(dots1, tags1, new_plot=True, subplot=221, title="原始点集")

    #利用sklearn库中的DBScan算法对dots做分类
    model1 = DBSCAN(eps=2, min_samples=1)
    db_pred_tags = model1.fit_predict(dots1)
    scatter_dots(dots1, db_pred_tags, new_plot=False, subplot=222, title="sklearn库中dbscan分类后的点集")

    #利用自己训练的dbscan算法最dots做分类
    mydb_pred_tags = my_dbscan(dots1, epsilon=2, minSamples=1)
    scatter_dots(dots1, mydb_pred_tags, new_plot=False, subplot=223, title="自己训练得到的dbscan分类后的点集")
    plt.show()

