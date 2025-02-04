import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración del backend
BASE_URL = "http://localhost:5000"  # Modificar según la dirección del backend
NUM_REPEATS = 10  # Número de repeticiones para promediar

# Parámetros de evaluación
n_components_range = range(1, 10)

# Diccionarios para almacenar resultados
results = {
    "GMM": {
        "bic": [], "aic": [], "silhouette": [], "ch_score": [], "db_score": [],
        "bic_std": [], "aic_std": [], "silhouette_std": [], "ch_score_std": [], "db_score_std": []
    },
    "KMeans": {
        "silhouette": [], "ch_score": [], "db_score": [],
        "silhouette_std": [], "ch_score_std": [], "db_score_std": []
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

# Evaluación de GMM y KMeans con múltiples random states
for n in n_components_range:
    gmm_bic_vals, gmm_aic_vals, gmm_sil_vals, gmm_ch_vals, gmm_db_vals = [], [], [], [], []
    kmeans_sil_vals, kmeans_ch_vals, kmeans_db_vals = [], [], []

    for _ in range(NUM_REPEATS):
        print(n_components_range)
        random_state = np.random.randint(0, 10000)

        # Ejecutar GMM
        gmm_payload = {
            "n_components": n, "covariance_type": "full",
            "n_samples": 300, "n_features": 2, "random_state": random_state
        }
        gmm_result = run_experiment("cluster", gmm_payload)
        if gmm_result:
            gmm_bic_vals.append(gmm_result.get("bic", None))
            gmm_aic_vals.append(gmm_result.get("aic", None))
            gmm_sil_vals.append(gmm_result.get("silhouette_score", None))
            gmm_ch_vals.append(gmm_result.get("calinski_harabasz_score", None))
            gmm_db_vals.append(gmm_result.get("davies_bouldin_score", None))

        # Ejecutar KMeans
        kmeans_payload = {
            "n_clusters": n, "n_samples": 300, "n_features": 2, "random_state": random_state
        }
        kmeans_result = run_experiment("kmeans", kmeans_payload)
        if kmeans_result:
            kmeans_sil_vals.append(kmeans_result.get("silhouette_score", None))
            kmeans_ch_vals.append(kmeans_result.get("calinski_harabasz_score", None))
            kmeans_db_vals.append(kmeans_result.get("davies_bouldin_score", None))

    # Función para calcular promedio y desviación estándar ignorando None
    def safe_mean(values):
        valid_values = [val for val in values if val is not None]
        return np.nanmean(valid_values) if valid_values else None

    def safe_std(values):
        valid_values = [val for val in values if val is not None]
        return np.nanstd(valid_values) if valid_values else None

    # Almacenar resultados promediados para GMM
    results["GMM"]["bic"].append(safe_mean(gmm_bic_vals))
    results["GMM"]["aic"].append(safe_mean(gmm_aic_vals))
    results["GMM"]["silhouette"].append(safe_mean(gmm_sil_vals))
    results["GMM"]["ch_score"].append(safe_mean(gmm_ch_vals))
    results["GMM"]["db_score"].append(safe_mean(gmm_db_vals))

    results["GMM"]["bic_std"].append(safe_std(gmm_bic_vals))
    results["GMM"]["aic_std"].append(safe_std(gmm_aic_vals))
    results["GMM"]["silhouette_std"].append(safe_std(gmm_sil_vals))
    results["GMM"]["ch_score_std"].append(safe_std(gmm_ch_vals))
    results["GMM"]["db_score_std"].append(safe_std(gmm_db_vals))

    # Almacenar resultados promediados para KMeans
    results["KMeans"]["silhouette"].append(safe_mean(kmeans_sil_vals))
    results["KMeans"]["ch_score"].append(safe_mean(kmeans_ch_vals))
    results["KMeans"]["db_score"].append(safe_mean(kmeans_db_vals))

    results["KMeans"]["silhouette_std"].append(safe_std(kmeans_sil_vals))
    results["KMeans"]["ch_score_std"].append(safe_std(kmeans_ch_vals))
    results["KMeans"]["db_score_std"].append(safe_std(kmeans_db_vals))

def plot_comparison(metric_name, ylabel, filename):
    plt.figure(figsize=(10, 5))

    # Convertir None a NaN antes de graficar
    gmm_mean = [np.nan if v is None else v for v in results["GMM"][metric_name]]
    gmm_std = [np.nan if v is None else v for v in results["GMM"][f"{metric_name}_std"]]

    kmeans_mean = [np.nan if v is None else v for v in results["KMeans"][metric_name]]
    kmeans_std = [np.nan if v is None else v for v in results["KMeans"][f"{metric_name}_std"]]

    # Gráfico de GMM
    plt.errorbar(n_components_range, gmm_mean, yerr=gmm_std, marker='o', linestyle='-', label="GMM", capsize=4)

    # Gráfico de KMeans (si aplica)
    if metric_name in results["KMeans"]:
        plt.errorbar(n_components_range, kmeans_mean, yerr=kmeans_std, marker='s', linestyle='--', label="KMeans", capsize=4)

    plt.xlabel("Número de Componentes")
    plt.ylabel(ylabel)
    plt.title(f"Comparación {ylabel}: GMM vs KMeans")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./graphs/{filename}')
    plt.close()


# Generar gráficos de comparación
plot_comparison("silhouette", "Silhouette Score", "comparison_silhouette.png")
plot_comparison("ch_score", "Calinski-Harabasz Score", "comparison_ch_score.png")
plot_comparison("db_score", "Davies-Bouldin Score", "comparison_db_score.png")

# Guardar resultados en un DataFrame
final_results = []
for i, n in enumerate(n_components_range):
    final_results.append([
        n, results["GMM"]["bic"][i], results["GMM"]["aic"][i], results["GMM"]["silhouette"][i],
        results["GMM"]["ch_score"][i], results["GMM"]["db_score"][i], results["KMeans"]["silhouette"][i],
        results["KMeans"]["ch_score"][i], results["KMeans"]["db_score"][i]
    ])

columns = ["N Components", "BIC (GMM)", "AIC (GMM)", "Silhouette (GMM)", "CH (GMM)", "DB (GMM)",
           "Silhouette (KMeans)", "CH (KMeans)", "DB (KMeans)"]
df_results = pd.DataFrame(final_results, columns=columns)

# Guardar en CSV
df_results.to_csv("comparison_gmm_vs_kmeans.csv", index=False)

print("Análisis completado. Resultados guardados en 'comparison_gmm_vs_kmeans.csv' y gráficos generados.")
