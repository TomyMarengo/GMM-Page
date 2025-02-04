import requests
import numpy as np
import matplotlib.pyplot as plt
import os

# Crear la carpeta graphs si no existe
os.makedirs("graphs", exist_ok=True)

# URL del backend (ajustar si es necesario)
BASE_URL = "http://localhost:5000/anomaly"

# Parámetros a variar
contamination_values = [0.01, 0.05, 0.1, 0.2]
n_components = 3  # Fijo en 3 para GMM
num_repeats = 10  # Número de repeticiones por configuración

# Almacenar resultados
metrics = {"GMM": {"f1": [], "precision": [], "recall": []}, "IsolationForest": {"f1": [], "precision": [], "recall": []}}
errors = {"GMM": {"f1": [], "precision": [], "recall": []}, "IsolationForest": {"f1": [], "precision": [], "recall": []}}

for contamination in contamination_values:
    temp_results = {"GMM": {"f1": [], "precision": [], "recall": []}, "IsolationForest": {"f1": [], "precision": [], "recall": []}}
    
    for repeat in range(num_repeats):
        print(contamination)
        random_state = np.random.randint(0, 10000)

        # Evaluar GMM
        gmm_payload = {
            "algorithm": "GMM",
            "contamination": contamination,
            "n_components": n_components,
            "random_state": random_state,
            "n_samples": 300,
            "n_features": 2
        }
        gmm_response = requests.post(BASE_URL, json=gmm_payload).json()
        
        # Evaluar Isolation Forest
        if_payload = {
            "algorithm": "IsolationForest",
            "contamination": contamination,
            "random_state": random_state,
            "n_samples": 300,
            "n_features": 2
        }
        if_response = requests.post(BASE_URL, json=if_payload).json()
        
        # Almacenar métricas temporales
        temp_results["GMM"]["f1"].append(gmm_response.get("metrics", {}).get("f1_score", 0))
        temp_results["GMM"]["precision"].append(gmm_response.get("metrics", {}).get("precision", 0))
        temp_results["GMM"]["recall"].append(gmm_response.get("metrics", {}).get("recall", 0))
        
        temp_results["IsolationForest"]["f1"].append(if_response.get("metrics", {}).get("f1_score", 0))
        temp_results["IsolationForest"]["precision"].append(if_response.get("metrics", {}).get("precision", 0))
        temp_results["IsolationForest"]["recall"].append(if_response.get("metrics", {}).get("recall", 0))

        # Seleccionar una corrida para graficar la distribución
        if repeat == 0:  # Solo la primera iteración de cada conjunto de parámetros
            def plot_anomaly_detection(data, labels, title, filename):
                """
                Genera un gráfico de dispersión con colores indicando anomalías.
                """
                data = np.array(data)
                labels = np.array(labels)
                
                plt.figure(figsize=(6, 6))
                plt.scatter(data[labels == 0, 0], data[labels == 0, 1], c='blue', label="Normal")
                plt.scatter(data[labels == 1, 0], data[labels == 1, 1], c='red', label="Anomalía")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.title(title)
                plt.legend()
                plt.grid()
                plt.savefig(filename)
                plt.close()

            # Graficar datos de GMM
            gmm_data = np.array(gmm_response["data"])
            gmm_labels = np.array(gmm_response["predictions"])
            plot_anomaly_detection(gmm_data, gmm_labels, 
                                   f"GMM - Contamination: {contamination}", 
                                   f"graphs/gmm_anomalies_cont_{contamination}.png")

            # Graficar datos de Isolation Forest
            if_data = np.array(if_response["data"])
            if_labels = np.array(if_response["predictions"])
            plot_anomaly_detection(if_data, if_labels, 
                                   f"Isolation Forest - Contamination: {contamination}", 
                                   f"graphs/if_anomalies_cont_{contamination}.png")

    # Calcular medias y desviaciones estándar
    for model in ["GMM", "IsolationForest"]:
        for metric in ["f1", "precision", "recall"]:
            metrics[model][metric].append(np.mean(temp_results[model][metric]))
            errors[model][metric].append(np.std(temp_results[model][metric]))

# Graficar comparación de métricas con barras de error en gráficos separados
metrics_names = ["f1", "precision", "recall"]
titles = ["F1-score", "Precision", "Recall"]

for metric, title in zip(metrics_names, titles):
    plt.figure(figsize=(8, 6))
    plt.errorbar(contamination_values, metrics["GMM"][metric], yerr=errors["GMM"][metric], marker='o', linestyle='-', label=f'GMM {metric}', capsize=5)
    plt.errorbar(contamination_values, metrics["IsolationForest"][metric], yerr=errors["IsolationForest"][metric], marker='s', linestyle='--', label=f'Isolation Forest {metric}', capsize=5)
    plt.xlabel('Contamination Level')
    plt.ylabel('Score')
    plt.title(f'Comparación de {title} entre GMM e Isolation Forest')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/outliers_{metric}_comparison.png')
    plt.close()

print("Análisis completado. Gráficos guardados en la carpeta 'graphs/'.")
