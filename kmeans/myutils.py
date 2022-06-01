import numpy as np
from sklearn import cluster


def get_random_samples(data, n):
    np.random.seed(1)
    indices = np.random.choice(len(data), size=n, replace=False)
    return data[indices, :]


def euclid_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def dist_sqaured(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_centers(data, clusters):
    centers = []

    l = np.unique(clusters)

    points = {}
    for i in l:
        points[i] = []

    for i in range(len(clusters)):
        points[clusters[i]].append(data[i])

    for i in l:
        total_x = 0
        total_y = 0

        for j in points[i]:
            total_x += j[0]
            total_y += j[1]
        centers.append([total_x / len(points[i]), total_y / len(points[i])])

    return np.array(centers)


def get_inertia(data, labels, centers):
    inertia = 0
    for i in range(data.shape[0]):
        inertia += dist_sqaured(data[i], centers[labels[i]])
    return inertia
