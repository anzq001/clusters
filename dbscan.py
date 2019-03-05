"""
聚类中的dbscan算法
DBSCAN算法是一种基于密度的聚类算法，算法的基本原理为
将一定密度中的元素包含在一个类中
"""
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["font.family"]= "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False
colors = ["r", "g", "b", "y", "m", "black", "orange", "gold"]
marker = ["o", "v", "*", "D", "d", ".", ",", "^"]

def generate_dots(dots_num):
    radiuses = random.choices([5, 10], k=dots_num)
    rads = random.choices(np.arange(0, 1, 0.01) * 2 * np.pi, k=dots_num)
    radiuses = radiuses + np.random.randn(dots_num) * 0.5
    rads = rads + np.random.randn(dots_num) * 0.01
    dots = radiuses * np.array([np.sin(rads), np.cos(rads)])
    return dots.T

# def isNeigh(center, dot, epsilon):
#     center = np.array(center, dtype=float)
#     dot = np.array(dot, dtype=float)
#     if np.sqrt(np.sum((center - dot) ** 2)) <= epsilon:
#         return True
#     else:
#         return False

def my_dbscan(dots, epsilon, minSamples):
    # v0.1版本，未考虑核心点和边界点的区别
    visit_points = [] # 将已访问的点的下标保存起来
    unvisit_points = [i for i in range(0, len(dots))] # 将未访问的点的下标保存起来
    tag = default_tag = -1 # 默认全为噪声点
    tags = [default_tag]  * len(dots)
    # 建立一个KD树，这种树采用类似平衡二叉树的算法，对多维数组建立二叉树，搜索时间复杂度为o(logn)
    kt = KDTree(dots)
    while unvisit_points.__len__() > 0:
        p_idx = unvisit_points[0]
        visit_points.append(p_idx)
        unvisit_points.remove(p_idx)
        neigh_idx = kt.query_ball_point(dots[p_idx], epsilon)
        if len(neigh_idx) > minSamples + 1:
            tag += 1
            tags[p_idx] = tag
            while neigh_idx.__len__() > 0:
                n_idx = neigh_idx[0]
                neigh_idx.remove(n_idx)
                if n_idx not in unvisit_points:
                    continue
                visit_points.append(n_idx)
                unvisit_points.remove(n_idx)
                next_neigh_idx = kt.query_ball_point(dots[n_idx], epsilon)
                if len(next_neigh_idx) > minSamples + 1:
                    neigh_idx = list(set(neigh_idx)|set(next_neigh_idx)&set(unvisit_points))
                tags[n_idx] = tag
        else:
            tags[p_idx] = -1
    return tags


if __name__ == "__main__":
    dots = generate_dots(1000)
    plt.figure(0)
    plt.subplot(221)
    plt.scatter(dots[:, 0], dots[:, 1], c="r", marker="o", s=2)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.title("原始图像")

    # eps: 同一类最大距离，min_sample: 核心点最小样本数, metric: 计算距离的方法，默认欧式距离，metric_param: 距离算法参数，
    # algorithm: 计算逐点距离和最近点的算法，leaf_size: 叶子的数量, p: 闵科夫斯基距离中的p, n_jobs: 线程数
    model = DBSCAN(eps=1.5, min_samples=5)
    pred_tag = model.fit_predict(dots)
    # pred_tag = [int(i) for i in pred_tag]
    plt.subplot(222)
    print(set(pred_tag))
    for index, dot in enumerate(dots):
        plt.scatter(dot[0], dot[1], c=colors[pred_tag[index]], marker=marker[pred_tag[index]], s=1)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.title("标准dbscan算法")

    my_pred_tags = my_dbscan(dots, epsilon=1.5, minSamples=5)
    plt.subplot(223)
    print(set(my_pred_tags))
    for index, dot in enumerate(dots):
        plt.scatter(dot[0], dot[1], c=colors[my_pred_tags[index]], marker=marker[my_pred_tags[index]], s=1)
    plt.xlim([-11, 11])
    plt.ylim([-11, 11])
    plt.title("自训练dbscan算法")
    plt.show()
