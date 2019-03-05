import numpy as np
import random
# kmeans算法

def compute_distance(dot, center):
    """
    采用欧式距离计算两点之间的距离
    :param dot: 点A(x1, y1)
    :param center: 点B(x2, y2)
    :return: sqrt((x1-x2)^2+(y1-y2)^2)
    """
    return np.sqrt(np.sum(np.power(np.array(dot) - np.array(center), 2)))


def compute_centers(class_num, dots, tags):
    """
    采用欧式距离计算类型中心点
    :param class_num: 类型数量
    :param dots: 二维点的集合
    :param tags: 点的标签
    :return: 返回类型中心点
    """
    dim = len(dots[0]) # 点的维度
    dots_num = np.zeros([class_num, 1], dtype=int)
    dots_sum = np.zeros([class_num, dim], dtype=float)
    for dot, tag in zip(dots, tags):
        dots_num[tag] += 1
        dots_sum[tag, :] = dots_sum[tag, :] + np.array(dot)
    new_centers = dots_sum/dots_num
    return new_centers.tolist()

def predict_tag(dot, centers):
    """
    根据点dot到centers的欧式距离，距离最近的即为dot的预测标签
    :param dot: type: tuple
    :param centers: type: list[tuple]
    :return: tag
    """
    dot = np.array(dot, dtype=float)
    centers = np.array(centers, dtype=float)
    distance = np.sqrt(np.sum(np.power(dot-centers, 2), axis=1))
    return np.argmin(distance)

def my_kmeans(dots, class_num):
    centers = random.sample(dots, class_num)
    predict_tags = []
    Iter = 0
    while Iter < 5:
        wcss = 0
        predict_tags.clear()
        for dot in dots:
            new_tag = predict_tag(dot, centers)
            predict_tags.append(new_tag)
            wcss += compute_distance(dot, centers[new_tag])
        centers = compute_centers(class_num, dots, predict_tags)
        Iter += 1
        print("第" + str(Iter) + "次训练，本次训练损失为" + str(wcss))
    # scatter_dots(centers, tags, size=5)
    return predict_tags