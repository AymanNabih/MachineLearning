import random
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from itertools import combinations, izip


def k_means(X, k=5, max_iter=1000):
    centers = [tuple(pt) for pt in random.sample(X, k)]
    for i in range(max_iter):
        clusters = defaultdict(list)

        # Expectation
        for datapoint in X:
            distances = []
            for center in centers:
                    distances.append(euclidean(datapoint, center))
            center = centers[np.argmin(distances)]
            clusters[center].append(datapoint)

        # Maximization
        new_centers = []
        for center, pts in clusters.iteritems():
            new_center = np.mean(pts, axis=0)
            new_centers.append(tuple(new_center))

        if set(new_centers) == set(centers):
            break

        centers = new_centers

    return clusters
    
def sse(clusters):
    return sum(euclidean(pt, center) ** 2 for center, pts in clusters.iteritems() for pt in pts)

def turn_clusters_into_labels(clusters):
    labels = []
    new_X = []
    label = 0
    for cluster, pts in clusters.iteritems():
        for pt in pts:
            new_X.append(pt)
            labels.append(label)
        label += 1
    return np.array(new_X), np.array(labels)


def plot_all_2d(X, feature_names, k=3):
    pairs = list(combinations(range(X.shape[1]), 2))
    fig, axes = plt.subplots((len(pairs) / 2), 2)
    flattened_axes = [ax for ls in axes for ax in ls]
    
    for pair, ax in izip(pairs, flattened_axes):
        pair = np.array(pair)
        plot_data_2d(X[:, pair], feature_names[pair], ax, k=k)
    plt.show()


def plot_data_2d(X, plot_labels, ax, k=3):
    clusters = k_means(X, k=k)
    new_X, labels = turn_clusters_into_labels(clusters)
    ax.scatter(new_X[:, 0], new_X[:, 1], c=labels)
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data

    plot_all_2d(X, np.array(iris.feature_names), k=5)
