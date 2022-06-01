import numpy as np
import matplotlib.pyplot as plt

from myutils import euclid_dist, get_centers, get_inertia, get_random_samples


data = np.genfromtxt('kmeans/gd.csv', delimiter=',')

KVALS = [2, 4, 6, 7]
INERTIA = []

def kmeans(data, k, max_iters=100):
    # random cluster centers
    cluster_centers = get_random_samples(data, k)

    # initially belong to same cluster
    clusters = np.zeros((data.shape[0],), dtype=np.int32)

    different = True
    while different and max_iters:
        for i in range(data.shape[0]):
            min_distance = float('inf')
            for j in range(k):
                cur_distance = euclid_dist(data[i], cluster_centers[j])
                if cur_distance < min_distance:
                    clusters[i] = j
                    min_distance = cur_distance
        new_cluster_centers = get_centers(data, clusters)

        if np.count_nonzero(cluster_centers-new_cluster_centers) == 0:
            different = False
        else:
            cluster_centers = new_cluster_centers
        max_iters -= 1

    return cluster_centers, clusters


for i, k in enumerate(KVALS):
    cluster_centers, cluster_labels = kmeans(data, k)

    INERTIA.append(get_inertia(data, cluster_labels, cluster_centers))

    plt.figure(i)
    plt.title('K-Means Clustering With K = ' + str(k))
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r')

plt.figure(len(KVALS))
plt.plot(KVALS, INERTIA, marker='o')
plt.title('Elbow Method')

print(INERTIA)

plt.show()
