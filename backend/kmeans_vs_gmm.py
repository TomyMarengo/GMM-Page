import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse
import os

# Crear la carpeta graphs si no existe
os.makedirs("graphs", exist_ok=True)

# Generar datos sintéticos con tres clusters bien definidos
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

X, y_true = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

### **1. Visualización de K-Means (Hard Clustering)**
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_kmeans, palette="deep", legend=False)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='x', label="Centroides")
plt.title("K-Means Clustering (Hard Clustering)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.savefig("graphs/kmeans_clustering.png")
plt.close()

### **2. Visualización de GMM (Soft Clustering con Gaussianas Solapadas)**
gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, covariance_type="full")
gmm.fit(X)
y_gmm = gmm.predict(X)
means = gmm.means_
covariances = gmm.covariances_

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_gmm, palette="deep", legend=False)

# Dibujar las elipses de las gaussianas
def draw_ellipse(position, covariance, ax, color):
    """Dibuja una elipse representando la distribución gaussiana"""
    if covariance.shape == (2, 2):  # Caso full
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)  # 2 std devs
        ellipse = Ellipse(xy=position, width=width, height=height, angle=angle, color=color, alpha=0.3)
        ax.add_patch(ellipse)

ax = plt.gca()
colors = sns.color_palette("deep", n_clusters)
for i, (mean, cov) in enumerate(zip(means, covariances)):
    draw_ellipse(mean, cov, ax, colors[i])

plt.title("GMM Clustering (Soft Clustering con Gaussianas Solapadas)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.savefig("graphs/gmm_clustering.png")
plt.close()
