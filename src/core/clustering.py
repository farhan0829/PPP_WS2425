import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ClusteringEngine:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def cluster(self, a_optimal):
        a_flat = a_optimal.flatten().reshape(-1, 1)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(a_flat)
        labels = kmeans.labels_.reshape(a_optimal.shape)
        return labels, kmeans.cluster_centers_

    def plot(self, a_optimal, labels):
        plt.imshow(labels, cmap='viridis')
        plt.colorbar(label='Cluster ID')
        plt.title("Clusters in Optimized a(x, y)")
        plt.show()