from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/cluster', methods=['POST'])
def cluster():
    """
    Endpoint para realizar clustering usando Gaussian Mixture Model.
    Si se recibe n_features mayor que 2, se aplica PCA para reducir a 2 componentes.
    """
    req_data = request.get_json()
    n_centers = req_data.get('n_centers', 3)       # Número de centros para blobs
    n_components = req_data.get('n_components', 3)         # Número de clusters para GMM
    random_state = req_data.get('random_state', 42)          # Semilla para reproducibilidad
    covariance_type = req_data.get('covariance_type', 'full')  # Tipo de covarianza
    n_samples = req_data.get('n_samples', 150)               # Número de puntos del dataset
    n_features = req_data.get('n_features', 2)               # Dimensiones del dataset

    # Generar un dataset con blobs
    data, labels_true = make_blobs(
        n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state
    )

    # Si se reciben más de 2 features, aplicar PCA para reducir a 2 componentes.
    if n_features > 2:
        pca = PCA(n_components=2, random_state=random_state)
        data = pca.fit_transform(data)
        n_features = 2  # Ahora trabajamos en 2 dimensiones

    # Ajustar el modelo GMM al dataset generado (en 2D si se aplicó PCA)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(data)

    clusters = gmm.predict(data).tolist()
    probabilities = gmm.predict_proba(data).tolist()
    means = gmm.means_.tolist()

    # Convertir las covarianzas a matrices cuadradas 2D
    raw_covariances = gmm.covariances_
    covariances = []

    if covariance_type == "diag":
        for cov in raw_covariances:
            cov_matrix = [[cov[i] if i == j else 0 for j in range(n_features)] for i in range(n_features)]
            covariances.append(cov_matrix)
    elif covariance_type == "spherical":
        for cov in raw_covariances:
            cov_matrix = [[cov if i == j else 0 for j in range(n_features)] for i in range(n_features)]
            covariances.append(cov_matrix)
    elif covariance_type == "tied":
        cov_matrix = raw_covariances.tolist() if hasattr(raw_covariances, "tolist") else raw_covariances
        covariances = [cov_matrix for _ in range(n_components)]
    else:  # full
        for cov in raw_covariances:
            cov_matrix = cov.tolist() if hasattr(cov, "tolist") else cov
            covariances.append(cov_matrix)

    score = gmm.score(data)
    bic = gmm.bic(data)
    aic = gmm.aic(data)

    # Calcular métricas de clustering
    try:
        sil_score = silhouette_score(data, clusters)
    except Exception:
        sil_score = None
    try:
        ch_score = calinski_harabasz_score(data, clusters)
    except Exception:
        ch_score = None
    try:
        db_score = davies_bouldin_score(data, clusters)
    except Exception:
        db_score = None
    try:
        ari = adjusted_rand_score(labels_true, clusters)
    except Exception:
        ari = None

    response = {
        "clusters": clusters,
        "probabilities": probabilities,
        "means": means,
        "covariances": covariances,
        "score": score,
        "bic": bic,
        "aic": aic,
        "n_iter": gmm.n_iter_,
        "converged": gmm.converged_,
        "data": data.tolist(),
        "labels_true": labels_true.tolist(),
        # Métricas adicionales
        "silhouette_score": sil_score,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score,
        "adjusted_rand_index": ari,
    }
    return jsonify(response)

@app.route('/kmeans', methods=['POST'])
def kmeans_cluster():
    """
    Endpoint para clustering usando KMeans.
    Si se recibe n_features mayor que 2, se aplica PCA para reducir a 2 componentes.
    """
    req_data = request.get_json()
    n_centers = req_data.get('n_centers', 3)
    n_clusters = req_data.get('n_clusters', 3)       # Número de clusters para KMeans
    random_state = req_data.get('random_state', 42)    # Semilla
    n_samples = req_data.get('n_samples', 150)         # Número de puntos
    n_features = req_data.get('n_features', 2)         # Dimensiones

    # Generar un dataset con blobs
    data, labels_true = make_blobs(
        n_samples=n_samples, centers=n_centers, n_features=n_features, random_state=random_state
    )

    # Si se reciben más de 2 features, aplicar PCA para reducir a 2 componentes.
    if n_features > 2:
        pca = PCA(n_components=2, random_state=random_state)
        data = pca.fit_transform(data)
        n_features = 2

    # Ajustar el modelo KMeans al dataset generado (en 2D si se aplicó PCA)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)

    clusters = kmeans.predict(data).tolist()
    centers = kmeans.cluster_centers_.tolist()
    inertia = kmeans.inertia_

    # Calcular métricas de clustering
    try:
        sil_score = silhouette_score(data, clusters)
    except Exception:
        sil_score = None
    try:
        ch_score = calinski_harabasz_score(data, clusters)
    except Exception:
        ch_score = None
    try:
        db_score = davies_bouldin_score(data, clusters)
    except Exception:
        db_score = None
    try:
        ari = adjusted_rand_score(labels_true, clusters)
    except Exception:
        ari = None

    response = {
        "clusters": clusters,
        "centers": centers,
        "inertia": inertia,
        "data": data.tolist(),
        "labels_true": labels_true.tolist(),
        # Métricas adicionales
        "silhouette_score": sil_score,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score,
        "adjusted_rand_index": ari,
    }
    return jsonify(response)

@app.route('/anomaly', methods=['POST'])
def anomaly_detection():
    """
    Endpoint para realizar detección de anomalías usando GMM o Isolation Forest.
    Si se recibe n_features mayor que 2, se aplica PCA para reducir a 2 componentes.
    """
    req_data = request.get_json()
    algorithm = req_data.get('algorithm', 'GMM')  # 'GMM' o 'IsolationForest'
    contamination = req_data.get('contamination', 0.05)  # Proporción de anomalías
    n_components = req_data.get('n_components', 3)  # Solo para GMM
    random_state = req_data.get('random_state', 42)
    n_samples = req_data.get('n_samples', 150)
    n_features = req_data.get('n_features', 2)

    # Generar dataset con blobs y algunas anomalías
    data, labels_true = make_blobs(
        n_samples=n_samples, centers=3, n_features=n_features, random_state=random_state
    )
    
    # Introducir anomalías
    n_anomalies = int(contamination * n_samples)
    np.random.seed(random_state)
    anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    data = np.vstack([data, anomalies])
    labels_true = np.hstack([labels_true, [-1]*n_anomalies])  # -1 para anomalías

    # Si se reciben más de 2 features, aplicar PCA para reducir a 2 componentes.
    if n_features > 2:
        pca = PCA(n_components=2, random_state=random_state)
        data = pca.fit_transform(data)
        n_features = 2

    if algorithm == 'GMM':
        model = GaussianMixture(n_components=n_components, random_state=random_state)
        model.fit(data)
        scores = model.score_samples(data)
        # Calcular umbral basado en la contaminación
        threshold = np.percentile(scores, contamination * 100)
        predictions = scores < threshold  # True para anomalías
    elif algorithm == 'IsolationForest':
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(data)
        predictions = model.predict(data)
        # Convertir predicciones: -1 para anomalías, 1 para normales
        predictions = predictions == -1
    else:
        return jsonify({"error": "Algoritmo no soportado"}), 400

    # Métricas de evaluación
    y_true = labels_true == -1
    y_pred = predictions

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, predictions)
    except:
        roc_auc = None

    response = {
        "algorithm": algorithm,
        "contamination": contamination,
        "predictions": predictions.tolist(),
        "data": data.tolist(),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        },
        "labels_true": labels_true.tolist(),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)