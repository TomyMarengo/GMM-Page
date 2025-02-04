import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuración del backend
BASE_URL = "http://localhost:5000"  # Modificar según la dirección del backend
NUM_REPEATS = 10  # Número de repeticiones para promediar

# Función para consumir la API y obtener resultados de GMM con distintos random states
def run_gmm_experiment(n_components, covariance_type="full", n_samples=300, n_features=2, random_state=None):
    payload = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "n_samples": n_samples,
        "n_features": n_features,
        "random_state": random_state
    }

    try:
        response = requests.post(f"{BASE_URL}/cluster", json=payload)
        
        if response.status_code != 200:
            print(f"Error: Código de respuesta {response.status_code}")
            print(f"Contenido de la respuesta: {response.text}")
            return None

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None

# Parámetros a evaluar
n_components_range = range(1, 10)
covariance_types = ["full", "diag", "tied", "spherical"]

# Diccionario para almacenar resultados con promedios y desviaciones estándar
results = {
    cov_type: {
        "bic": [], "aic": [], "silhouette": [], "ch_score": [], "db_score": [],
        "bic_std": [], "aic_std": [], "silhouette_std": [], "ch_score_std": [], "db_score_std": []
    }
    for cov_type in covariance_types
}

# Evaluación de GMM con múltiples random states
for cov_type in covariance_types:
    for n in n_components_range:
        bic_vals, aic_vals, silhouette_vals, ch_vals, db_vals = [], [], [], [], []
        
        for _ in range(NUM_REPEATS):
            print("ok")
            random_state = np.random.randint(0, 10000)  # Generar un random state diferente en cada iteración
            result = run_gmm_experiment(n, covariance_type=cov_type, random_state=random_state)
            
            if result:
                bic_vals.append(result["bic"])
                aic_vals.append(result["aic"])
                silhouette_vals.append(result.get("silhouette_score", None))
                ch_vals.append(result.get("calinski_harabasz_score", None))
                db_vals.append(result.get("davies_bouldin_score", None))

        # Filtrar valores None antes de calcular la media
        def safe_mean(values):
            valid_values = [val for val in values if val is not None]
            return np.nanmean(valid_values) if valid_values else None
        
        def safe_std(values):
            valid_values = [val for val in values if val is not None]
            return np.nanstd(valid_values) if valid_values else None

        results[cov_type]["bic"].append(safe_mean(bic_vals))
        results[cov_type]["aic"].append(safe_mean(aic_vals))
        results[cov_type]["silhouette"].append(safe_mean(silhouette_vals))
        results[cov_type]["ch_score"].append(safe_mean(ch_vals))
        results[cov_type]["db_score"].append(safe_mean(db_vals))

        results[cov_type]["bic_std"].append(safe_std(bic_vals))
        results[cov_type]["aic_std"].append(safe_std(aic_vals))
        results[cov_type]["silhouette_std"].append(safe_std(silhouette_vals))
        results[cov_type]["ch_score_std"].append(safe_std(ch_vals))
        results[cov_type]["db_score_std"].append(safe_std(db_vals))

def plot_metric_with_error_bars(metric_name, ylabel, filename):
    plt.figure(figsize=(10, 5))
    
    for cov_type in covariance_types:
        mean_vals = results[cov_type][metric_name]
        std_vals = results[cov_type][f"{metric_name}_std"]

        # Convertir None a NaN
        mean_vals = [np.nan if v is None else v for v in mean_vals]
        std_vals = [np.nan if v is None else v for v in std_vals]

        plt.errorbar(n_components_range, mean_vals, yerr=std_vals, marker='o', label=f"{cov_type}", capsize=4)

    plt.xlabel("Número de Componentes")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} en función del número de componentes")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./graphs/{filename}')
    plt.close()

# Generar gráficos con barras de error para todas las métricas
plot_metric_with_error_bars("bic", "BIC", "bic_comparison.png")
plot_metric_with_error_bars("aic", "AIC", "aic_comparison.png")
plot_metric_with_error_bars("silhouette", "Silhouette Score", "silhouette_comparison.png")
plot_metric_with_error_bars("ch_score", "Calinski-Harabasz Score", "ch_score_comparison.png")
plot_metric_with_error_bars("db_score", "Davies-Bouldin Score", "db_score_comparison.png")

# Guardar resultados en un DataFrame
final_results = []
for cov_type in covariance_types:
    for i, n in enumerate(n_components_range):
        final_results.append([
            cov_type, n, results[cov_type]["bic"][i], results[cov_type]["aic"][i],
            results[cov_type]["silhouette"][i], results[cov_type]["ch_score"][i], results[cov_type]["db_score"][i],
            results[cov_type]["bic_std"][i], results[cov_type]["aic_std"][i],
            results[cov_type]["silhouette_std"][i], results[cov_type]["ch_score_std"][i], results[cov_type]["db_score_std"][i]
        ])

columns = [
    "Covariance Type", "N Components", "BIC", "AIC", "Silhouette Score", 
    "Calinski-Harabasz", "Davies-Bouldin", "BIC Std", "AIC Std", 
    "Silhouette Std", "Calinski-Harabasz Std", "Davies-Bouldin Std"
]
df_results = pd.DataFrame(final_results, columns=columns)

# Guardar en CSV
df_results.to_csv("gmm_results.csv", index=False)

print("Análisis completado. Resultados guardados en 'gmm_results.csv' y gráficos generados.")
