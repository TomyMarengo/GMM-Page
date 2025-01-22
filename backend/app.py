from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

app = Flask(__name__)
CORS(app)

# Cargamos el dataset Iris una sola vez al iniciar la aplicación.
iris = datasets.load_iris()
data = iris.data
feature_names = iris.feature_names

@app.route('/cluster', methods=['POST'])
def cluster():
    """
    Endpoint para realizar clustering usando Gaussian Mixture Model.
    """
    req_data = request.get_json()
    n_components = req_data.get('n_components', 3)  # Número de clusters para GMM
    random_state = req_data.get('random_state', 42)  # Semilla para reproducibilidad
    covariance_type = req_data.get('covariance_type', 'full')  # Tipo de covarianza
    n_samples = req_data.get('n_samples', 150)  # Número de puntos del dataset
    n_features = req_data.get('n_features', 2)  # Dimensiones del dataset

    # Generar un dataset con blobs
    data, labels_true = make_blobs(n_samples=n_samples, centers=n_components, n_features=n_features, random_state=random_state)

    # Ajustar el modelo GMM al dataset generado
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type=covariance_type)
    gmm.fit(data)

    clusters = gmm.predict(data).tolist()
    probabilities = gmm.predict_proba(data).tolist()
    means = gmm.means_.tolist()

    # Convertir las covarianzas a matrices cuadradas 2D
    raw_covariances = gmm.covariances_
    covariances = []

    if covariance_type == "diag":
        # Cada componente retorna un vector con n_features elementos
        for cov in raw_covariances:
            cov_matrix = [[cov[i] if i == j else 0 for j in range(n_features)] for i in range(n_features)]
            covariances.append(cov_matrix)
    elif covariance_type == "spherical":
        # Cada componente retorna un escalar
        for cov in raw_covariances:
            cov_matrix = [[cov if i == j else 0 for j in range(n_features)] for i in range(n_features)]
            covariances.append(cov_matrix)
    elif covariance_type == "tied":
        # raw_covariances es una única matriz que se comparte para todos los clusters,
        # así que la replicamos para cada componente.
        cov_matrix = raw_covariances.tolist() if hasattr(raw_covariances, "tolist") else raw_covariances
        covariances = [cov_matrix for _ in range(n_components)]
    else:
        # full: cada componente ya tiene su propia matriz (suponemos que ya es cuadrada)
        for cov in raw_covariances:
            cov_matrix = cov.tolist() if hasattr(cov, "tolist") else cov
            covariances.append(cov_matrix)

    score = gmm.score(data)
    bic = gmm.bic(data)
    aic = gmm.aic(data)

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
        "labels_true": labels_true.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
