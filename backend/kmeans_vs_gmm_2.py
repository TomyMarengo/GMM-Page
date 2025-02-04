import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import os

os.makedirs("graphs", exist_ok=True)

# Función para graficar clustering de K-Means
def plot_kmeans(ax, data, labels, centers, title):
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label="Centroides")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Función para graficar clustering de GMM con elipses de covarianza
def plot_gmm(ax, data, labels, gmm, title):
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20)
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        if cov.shape == (2, 2):  
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.arctan2(*eigenvectors[:, 0][::-1])
            angle = np.degrees(angle)
            width, height = 2.5 * np.sqrt(eigenvalues)  # Aumentar tamaño de las elipses
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='black', facecolor='none', lw=2)
            ax.add_patch(ellipse)
    ax.set_title(title)
    ax.grid(True)

# **Generar dataset con clusters esféricos**
n_samples = 300
X_spherical, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)

# **Generar dataset con clusters elípticos y solapamiento**
X_elliptical, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=2.0, random_state=42)  # Aumentar dispersión
transformation_matrix = np.array([[0.8, -0.6], [1.5, 0.8]])  # Más deformación para mayor solapamiento
X_elliptical = X_elliptical @ transformation_matrix

# **Aplicar K-Means y GMM**
kmeans_spherical = KMeans(n_clusters=3, random_state=42).fit(X_spherical)
gmm_spherical = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(X_spherical)

kmeans_elliptical = KMeans(n_clusters=3, random_state=42).fit(X_elliptical)
gmm_elliptical = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(X_elliptical)

# **Graficar resultados**
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_kmeans(axes[0, 0], X_spherical, kmeans_spherical.labels_, kmeans_spherical.cluster_centers_, "K-Means - Clusters Esféricos")
plot_gmm(axes[0, 1], X_spherical, gmm_spherical.predict(X_spherical), gmm_spherical, "GMM - Clusters Esféricos")

plot_kmeans(axes[1, 0], X_elliptical, kmeans_elliptical.labels_, kmeans_elliptical.cluster_centers_, "K-Means - Clusters Solapados")
plot_gmm(axes[1, 1], X_elliptical, gmm_elliptical.predict(X_elliptical), gmm_elliptical, "GMM - Clusters Solapados")

plt.tight_layout()
plt.savefig("graphs/kmeans_vs_gmm_comparison.png")
plt.close()
