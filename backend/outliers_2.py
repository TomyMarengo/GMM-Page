import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Crear carpeta si no existe
os.makedirs("graphs", exist_ok=True)

# Configuración del backend
BASE_URL = "http://localhost:5000/anomaly"
contamination_values = [0.01, 0.05, 0.1, 0.2]

def get_predictions(algorithm, contamination):
    """Solicita predicciones al backend."""
    payload = {
        "algorithm": algorithm,
        "contamination": contamination,
        "n_samples": 300,
        "n_features": 2,
        "random_state": np.random.randint(0, 10000)
    }
    response = requests.post(BASE_URL, json=payload).json()
    print(response)
    return np.array(response["predictions"]), np.array(response["labels_true"])

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Genera la matriz de confusión para cada modelo."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalía"], yticklabels=["Normal", "Anomalía"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title(title)
    plt.savefig(filename)
    plt.close()

for contamination in contamination_values:
    gmm_preds, y_true = get_predictions("GMM", contamination)
    if_preds, _ = get_predictions("IsolationForest", contamination)

    plot_confusion_matrix(y_true, gmm_preds, f"Matriz de Confusión - GMM (cont={contamination})", f"graphs/confusion_gmm_{contamination}.png")
    plot_confusion_matrix(y_true, if_preds, f"Matriz de Confusión - Isolation Forest (cont={contamination})", f"graphs/confusion_if_{contamination}.png")

def get_anomaly_scores(algorithm, contamination):
    """Obtiene los scores de anomalía desde el backend."""
    payload = {
        "algorithm": algorithm,
        "contamination": contamination,
        "n_samples": 300,
        "n_features": 2,
        "random_state": np.random.randint(0, 10000)
    }
    response = requests.post(BASE_URL, json=payload).json()
    return np.array(response["scores"])

for contamination in contamination_values:
    gmm_scores = get_anomaly_scores("GMM", contamination)
    if_scores = get_anomaly_scores("IsolationForest", contamination)

    plt.figure(figsize=(8, 6))
    plt.boxplot([gmm_scores, if_scores], labels=["GMM", "Isolation Forest"], patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.xlabel("Modelo")
    plt.ylabel("Score de Anomalía")
    plt.title(f"Distribución de Scores de Anomalía (cont={contamination})")
    plt.savefig(f"./graphs/boxplot_anomaly_scores_{contamination}.png")
    plt.close()

from sklearn.metrics import roc_curve, auc

def get_probabilities(algorithm, contamination):
    """Solicita probabilidades de anomalía al backend."""
    payload = {
        "algorithm": algorithm,
        "contamination": contamination,
        "n_samples": 300,
        "n_features": 2,
        "random_state": np.random.randint(0, 10000)
    }
    response = requests.post(BASE_URL, json=payload).json()
    return np.array(response["scores"]), np.array(response["labels_true"])

for contamination in contamination_values:
    gmm_probs, y_true = get_probabilities("GMM", contamination)
    if_probs, _ = get_probabilities("IsolationForest", contamination)


    # Asegurar que y_true es binario (0 = normal, 1 = anomalía)
    y_true = (y_true != 0).astype(int)  

    # Calcular la curva ROC
    fpr_gmm, tpr_gmm, _ = roc_curve(y_true, gmm_probs)
    # Calcular curvas ROC
    fpr_gmm, tpr_gmm, _ = roc_curve(y_true, gmm_probs)
    fpr_if, tpr_if, _ = roc_curve(y_true, if_probs)
    auc_gmm = auc(fpr_gmm, tpr_gmm)
    auc_if = auc(fpr_if, tpr_if)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gmm, tpr_gmm, label=f"GMM (AUC = {auc_gmm:.2f})", linestyle='-')
    plt.plot(fpr_if, tpr_if, label=f"Isolation Forest (AUC = {auc_if:.2f})", linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal referencia
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curvas ROC - Contamination {contamination}")
    plt.legend()
    plt.savefig(f"./graphs/roc_curve_{contamination}.png")
    plt.close()
