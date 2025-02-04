import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuración del endpoint
url = "http://localhost:5000/density"
datasets = ["iris", "housing"]  # Diferentes datasets a evaluar
models = {"GMM": "density", "KMeans": "kmeans", "IsolationForest": "isolation_forest"}

# Diccionario para almacenar métricas
metrics_results = {"model": [], "dataset": [], "silhouette": [], "davies_bouldin": []}

for dataset in datasets:
    params = {
        "n_components": 3,
        "random_state": 42,
        "covariance_type": "full",
        "dataset": dataset
    }
    
    # Enviar solicitud al backend
    response = requests.post(url, json=params)
    
    if response.status_code == 200:
        data = response.json()
        X = np.array(data["data"])
        
        # GMM - Evaluación y métricas
        gmm_labels = np.argmax(np.array(data["probabilities"]), axis=1)
        silhouette_gmm = silhouette_score(X, gmm_labels)
        davies_bouldin_gmm = davies_bouldin_score(X, gmm_labels)
        metrics_results["model"].append("GMM")
        metrics_results["dataset"].append(dataset)
        metrics_results["silhouette"].append(silhouette_gmm)
        metrics_results["davies_bouldin"].append(davies_bouldin_gmm)
        
        # KMeans - Evaluación y métricas
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        silhouette_kmeans = silhouette_score(X, kmeans.labels_)
        davies_bouldin_kmeans = davies_bouldin_score(X, kmeans.labels_)
        metrics_results["model"].append("KMeans")
        metrics_results["dataset"].append(dataset)
        metrics_results["silhouette"].append(silhouette_kmeans)
        metrics_results["davies_bouldin"].append(davies_bouldin_kmeans)
        
        # Isolation Forest - Evaluación y métricas (sin clustering, usando scores)
        iso_forest = IsolationForest(random_state=42).fit(X)
        anomaly_scores = iso_forest.decision_function(X)
        labels_if = (anomaly_scores < np.percentile(anomaly_scores, 10)).astype(int)  # Definir anomalías como un pseudo-cluster
        silhouette_if = silhouette_score(X, labels_if)
        davies_bouldin_if = davies_bouldin_score(X, labels_if)
        metrics_results["model"].append("IsolationForest")
        metrics_results["dataset"].append(dataset)
        metrics_results["silhouette"].append(silhouette_if)
        metrics_results["davies_bouldin"].append(davies_bouldin_if)

# Graficar comparación de métricas
for dataset in datasets:
    for metric in ["silhouette", "davies_bouldin"]:
        plt.figure(figsize=(8, 5))
        model_values = [metrics_results[metric][i] for i in range(len(metrics_results["dataset"])) if metrics_results["dataset"][i] == dataset]
        models_labels = [metrics_results["model"][i] for i in range(len(metrics_results["dataset"])) if metrics_results["dataset"][i] == dataset]
        plt.plot(models_labels, model_values, marker='o', linestyle='-', label=f'{dataset}')
        plt.xlabel("Modelo")
        plt.ylabel(metric)
        plt.title(f'Comparación de {metric} entre Modelos - {dataset}')
        plt.legend()
        plt.savefig(f'model_comparison_{metric}_{dataset}.png')
        plt.close()
