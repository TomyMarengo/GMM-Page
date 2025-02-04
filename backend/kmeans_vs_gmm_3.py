import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración del backend
BASE_URL = "http://localhost:5000"  # Modificar según la dirección del backend
NUM_REPEATS = 10  # Número de repeticiones para promediar

# Parámetros de evaluación
n_components_range = range(2, 10)  # Se comienza en 2 para evitar clusters triviales

# Diccionarios para almacenar resultados
datasets = {
    "Clusters Esféricos": {
        "endpoint": "spherical_clusters",  # Asegurar que este endpoint esté en el backend
        "results": {
            "GMM": {"silhouette": [], "silhouette_std": []},
            "KMeans": {"silhouette": [], "silhouette_std": []}
        }
    },
    "Clusters Solapados": {
        "endpoint": "overlapping_clusters",  # Asegurar que este endpoint esté en el backend
        "results": {
            "GMM": {"silhouette": [], "silhouette_std": []},
            "KMeans": {"silhouette": [], "silhouette_std": []}
        }
    }
}

# Función para consumir la API y obtener resultados de clustering
def run_experiment(endpoint, payload):
    try:
        response = requests.post(f"{BASE_URL}/{endpoint}", json=payload)
        if response.status_code != 200:
            print(f"Error en {endpoint}: Código {response.status_code}, Respuesta: {response.text}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud a {endpoint}: {e}")
        return None

# Evaluación de GMM y KMeans en ambos datasets
for dataset_name, dataset_info in datasets.items():
    endpoint = dataset_info["endpoint"]
    results = dataset_info["results"]

    for n in n_components_range:
        gmm_sil_vals, kmeans_sil_vals = [], []

        for _ in range(NUM_REPEATS):
            print(n)
            random_state = np.random.randint(0, 10000)

            # Ejecutar GMM
            gmm_payload = {
                "n_components": n, "covariance_type": "full",
                "n_samples": 300, "n_features": 2, "random_state": random_state
            }
            gmm_result = run_experiment("cluster", gmm_payload)
            if gmm_result:
                gmm_sil_vals.append(gmm_result.get("silhouette_score", None))

            # Ejecutar KMeans
            kmeans_payload = {
                "n_clusters": n, "n_samples": 300, "n_features": 2, "random_state": random_state
            }
            kmeans_result = run_experiment("kmeans", kmeans_payload)
            if kmeans_result:
                kmeans_sil_vals.append(kmeans_result.get("silhouette_score", None))

        # Función para calcular promedio y desviación estándar ignorando None
        def safe_mean(values):
            valid_values = [val for val in values if val is not None]
            return np.nanmean(valid_values) if valid_values else None

        def safe_std(values):
            valid_values = [val for val in values if val is not None]
            return np.nanstd(valid_values) if valid_values else None

        # Almacenar resultados promediados para GMM
        results["GMM"]["silhouette"].append(safe_mean(gmm_sil_vals))
        results["GMM"]["silhouette_std"].append(safe_std(gmm_sil_vals))

        # Almacenar resultados promediados para KMeans
        results["KMeans"]["silhouette"].append(safe_mean(kmeans_sil_vals))
        results["KMeans"]["silhouette_std"].append(safe_std(kmeans_sil_vals))

# Función para graficar la comparación
def plot_comparison(dataset_name, results, filename):
    plt.figure(figsize=(10, 5))

    gmm_mean = [np.nan if v is None else v for v in results["GMM"]["silhouette"]]
    gmm_std = [np.nan if v is None else v for v in results["GMM"]["silhouette_std"]]

    kmeans_mean = [np.nan if v is None else v for v in results["KMeans"]["silhouette"]]
    kmeans_std = [np.nan if v is None else v for v in results["KMeans"]["silhouette_std"]]

    plt.errorbar(n_components_range, gmm_mean, yerr=gmm_std, marker='o', linestyle='-', label="GMM", capsize=4)
    plt.errorbar(n_components_range, kmeans_mean, yerr=kmeans_std, marker='s', linestyle='--', label="KMeans", capsize=4)

    plt.xlabel("Número de Componentes")
    plt.ylabel("Silhouette Score")
    plt.title(f"Comparación de Silhouette Score: GMM vs KMeans en {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./graphs/{filename}')
    plt.close()

# Generar gráficos para cada dataset
for dataset_name, dataset_info in datasets.items():
    plot_comparison(dataset_name, dataset_info["results"], f"comparison_silhouette_{dataset_name.replace(' ', '_')}.png")

print("Análisis completado. Gráficos generados en la carpeta './graphs/'.")
