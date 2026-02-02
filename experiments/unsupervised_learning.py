"""
Unsupervised Learning Examples

Unsupervised learning finds patterns in data without labeled targets.
This script demonstrates:
  1. K-Means clustering — group similar data points into clusters
  2. Principal Component Analysis (PCA) — reduce dimensionality and visualize

Requirements: numpy, matplotlib, scikit-learn
Install: pip install numpy matplotlib scikit-learn
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


def run_kmeans_demo():
    """
    Generate synthetic data with clear clusters and apply K-Means.
    Visualize the clusters and cluster centers.
    """
    # Generate 300 samples in 2D with 3 natural clusters
    np.random.seed(42)
    X, true_labels = make_blobs(n_samples=300, n_features=2, centers=3,
                                cluster_std=0.8, random_state=42)

    # Fit K-Means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # True labels (ground truth — usually unknown in real unsupervised setting)
    ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap="viridis", alpha=0.7, edgecolors="k")
    ax1.set_title("True clusters (synthetic)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    # K-Means result
    ax2.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7, edgecolors="k")
    ax2.scatter(centers[:, 0], centers[:, 1], c="red", marker="X", s=200, label="Centroids")
    ax2.set_title("K-Means clustering (k=3)")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("kmeans_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("K-Means: clusters assigned. Centroids:", centers)


def run_pca_demo():
    """
    Generate higher-dimensional data and use PCA to reduce to 2D for visualization.
    """
    # Data in 5 dimensions
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, n_features=5, centers=3, cluster_std=1.2, random_state=42)

    # Reduce to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Variance explained by each component
    var_ratio = pca.explained_variance_ratio_
    print(f"PCA: variance explained by PC1: {var_ratio[0]:.2%}, PC2: {var_ratio[1]:.2%}")

    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="steelblue", alpha=0.7, edgecolors="k")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.title("Data in 2D after PCA (from 5D)")
    plt.tight_layout()
    plt.savefig("pca_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=== Unsupervised Learning Examples ===\n")
    run_kmeans_demo()
    print()
    run_pca_demo()
