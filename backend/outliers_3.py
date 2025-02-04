import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_blobs

# Parámetros
contamination = 0.05
n_components = 3
n_samples = 300
n_features = 2
random_state = 42

# Generar datos sintéticos
data, labels_true = make_blobs(n_samples=n_samples, centers=n_components, n_features=n_features, random_state=random_state)

# Generar anomalías
def generate_anomalies(n_anomalies, n_features):
    return np.random.uniform(low=-10, high=20, size=(n_anomalies, n_features))

n_anomalies = int(contamination * n_samples)
anomalies = generate_anomalies(n_anomalies, n_features)

data = np.vstack([data, anomalies])
labels_true = np.hstack([labels_true, [-1] * n_anomalies])

# Aplicar escalado
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# GMM con datos escalados
gmm = GaussianMixture(n_components=n_components, random_state=random_state)
gmm.fit(data_scaled)
scores_gmm = gmm.score_samples(data_scaled)
threshold_gmm = np.percentile(scores_gmm, contamination * 100)
predictions_gmm = scores_gmm < threshold_gmm

# Isolation Forest con datos escalados
iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
iso_forest.fit(data_scaled)
scores_if = iso_forest.decision_function(data_scaled)
predictions_if = iso_forest.predict(data_scaled) == -1

# Calcular Curva ROC
y_true = (labels_true == -1)
fpr_gmm, tpr_gmm, _ = roc_curve(y_true, -scores_gmm)
fpr_if, tpr_if, _ = roc_curve(y_true, -scores_if)
auc_gmm = auc(fpr_gmm, tpr_gmm)
auc_if = auc(fpr_if, tpr_if)

# Graficar Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_gmm, tpr_gmm, label=f'GMM (AUC = {auc_gmm:.2f})', linestyle='-')
plt.plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {auc_if:.2f})', linestyle='--')
plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Curvas ROC - Contamination {contamination} (Datos Escalados)')
plt.legend()
plt.grid()
plt.show()