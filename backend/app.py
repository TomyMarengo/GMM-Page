from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml
from skimage import io, color
from PIL import Image
from skimage.transform import resize
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
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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
    
    # Parámetros adicionales para Isolation Forest
    n_estimators = req_data.get('n_estimators', 100)  # Solo para Isolation Forest
    max_samples = req_data.get('max_samples', 'auto')  # Solo para Isolation Forest

    if isinstance(max_samples, str):
        if max_samples.lower() == 'auto':
            max_samples = 'auto'
        else:
            try:
                if '.' in max_samples:
                    max_samples = float(max_samples)
                else:
                    max_samples = int(max_samples)
            except ValueError:
                return jsonify({"error": "max_samples debe ser 'auto' o un número válido"}), 400
    elif isinstance(max_samples, (int, float)):
        pass
    else:
        return jsonify({"error": "max_samples debe ser 'auto' o un número válido"}), 400
    
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
        scores = model.score_samples(data)  # Puntuaciones de GMM
        # Calcular umbral basado en la contaminación
        threshold = np.percentile(scores, contamination * 100)
        predictions = scores < threshold  # True para anomalías
    elif algorithm == 'IsolationForest':
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples
        )
        model.fit(data)
        scores = model.decision_function(data)  # Puntuaciones de Isolation Forest
        predictions = model.predict(data) == -1  # True para anomalías
    else:
        return jsonify({"error": "Algoritmo no soportado"}), 400
    
    # Métricas de evaluación
    y_true = labels_true == -1
    y_pred = predictions
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, scores if algorithm == 'GMM' else -scores)  # Invertir scores para Isolation Forest
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
        "scores": scores.tolist(),  # Añadido
    }
    
    return jsonify(response)

@app.route('/density', methods=['POST'])
def density_modeling():
    """
    Endpoint para realizar modelado de densidad usando Gaussian Mixture Model.
    Acepta un dataset como entrada, analiza la densidad y retorna visualización 2D.
    """
    req_data = request.get_json()
    n_components = req_data.get('n_components', 3)  # Número de componentes GMM
    random_state = req_data.get('random_state', 42)  # Semilla para reproducibilidad
    covariance_type = req_data.get('covariance_type', 'full')  # Tipo de covarianza
    dataset_name = req_data.get('dataset', 'iris')  # Dataset para análisis

    # Seleccionar dataset
    if dataset_name == 'iris':
        from sklearn.datasets import load_iris
        dataset = load_iris(as_frame=True)
        data = dataset.frame
        feature_names = dataset.feature_names
    elif dataset_name == 'housing':
        from sklearn.datasets import fetch_california_housing
        dataset = fetch_california_housing(as_frame=True)
        data = dataset.frame
        feature_names = dataset.feature_names
    else:
        return jsonify({"error": "Dataset no soportado"}), 400

    # Seleccionar solo variables numéricas
    features = data.select_dtypes(include=['float64', 'int64']).values

    # Reducir dimensiones si hay más de 2 features
    if features.shape[1] > 2:
        pca = PCA(n_components=2, random_state=random_state)
        features = pca.fit_transform(features)

    # Ajustar el modelo GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(features)

    density = np.exp(gmm.score_samples(features))  # Calcular la densidad
    probabilities = gmm.predict_proba(features).tolist()

    # Respuesta JSON
    response = {
        "data": features.tolist(),
        "density": density.tolist(),
        "probabilities": probabilities,
        "means": gmm.means_.tolist(),
        "covariances": gmm.covariances_.tolist(),
        "bic": gmm.bic(features),
        "aic": gmm.aic(features),
        "converged": gmm.converged_,
        "feature_names": feature_names,
    }
    return jsonify(response)

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('./', filename)

@app.route('/datasets/mnist', methods=['GET'])
def get_mnist_images():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    dataset_dir = "./datasets/mnist"
    os.makedirs(dataset_dir, exist_ok=True)

    image_files = []
    for i, img_array in enumerate(x_train[:20]):
        img_path = os.path.join(dataset_dir, f"mnist_{i}.png")
        img = Image.fromarray(img_array)
        img.save(img_path)
        image_files.append({
            "url": f"http://localhost:5000/datasets/mnist/mnist_{i}.png",
            "label": int(y_train[i])
        })

    return jsonify(image_files)

@app.route('/datasets/cifar10', methods=['GET'])
def get_cifar10_images():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), _ = cifar10.load_data()

    dataset_dir = "./datasets/cifar10"
    os.makedirs(dataset_dir, exist_ok=True)

    image_files = []
    for i, img_array in enumerate(x_train[:20]):  # Limitamos a 20 imágenes para demo
        img_path = os.path.join(dataset_dir, f"cifar10_{i}.png")
        img = Image.fromarray(img_array)
        img.save(img_path)
        image_files.append({
            "url": f"http://localhost:5000/datasets/cifar10/cifar10_{i}.png",
            "label": int(y_train[i][0])
        })
    
    return jsonify(image_files)

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Endpoint para segmentar imágenes usando GMM o KMeans.
    """
    req_data = request.get_json()
    algorithm = req_data.get('algorithm', 'GMM')  # GMM o KMeans
    n_components = req_data.get('n_components', 3)  # Número de clusters
    image_url = req_data.get('image_url')          # URL de la imagen
    resize_shape = req_data.get('resize_shape', (256, 256))  # Tamaño de redimensionado

    if not image_url:
        return jsonify({"error": "Se requiere una URL de la imagen"}), 400

    # Cargar y procesar la imagen
    # Cargar y procesar la imagen
    try:
        image = io.imread(image_url)
        image = resize(image, resize_shape, anti_aliasing=True)

        # Asegurarse de que la imagen tenga 3 canales (RGB)
        if len(image.shape) == 2:  # Escala de grises
            image = np.stack((image,)*3, axis=-1)

        image_lab = color.rgb2lab(image)  # Convertir a LAB para mejor segmentación
        flat_image = image_lab.reshape((-1, 3))
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 400

    if algorithm == 'GMM':
        model = GaussianMixture(n_components=n_components, random_state=42)
        model.fit(flat_image)
        labels = model.predict(flat_image)
        cluster_centers = model.means_
    elif algorithm == 'KMeans':
        model = KMeans(n_clusters=n_components, random_state=42)
        model.fit(flat_image)
        labels = model.labels_
        cluster_centers = model.cluster_centers_
    else:
        return jsonify({"error": "Algoritmo no soportado. Use 'GMM' o 'KMeans'."}), 400

    # Reconstruir la imagen segmentada
    segmented_image = cluster_centers[labels].reshape(image_lab.shape)
    segmented_image_rgb = color.lab2rgb(segmented_image)

    response = {
        "segmented_image": segmented_image_rgb.tolist(),
        "labels": labels.reshape(resize_shape[:2]).tolist(),
        "cluster_centers": cluster_centers.tolist(),
        "algorithm": algorithm
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)