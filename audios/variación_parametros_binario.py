import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Directorio donde están los archivos de audio
audio_dir = "./audios"

# Lista de valores 'xx' válidos
valid_xx = ['01', '02', '03', '04', '05', '06', '07', '08']

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extracción de MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Energía (Root Mean Square)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    
    # Chroma (Tonalidad)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    return np.hstack([np.mean(mfccs, axis=1), np.mean(zcr, axis=1), energy, np.mean(chroma, axis=1)])

# Recorrer archivos y extraer características
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
features = []
file_names = []

for file in audio_files:
    xx = file.split('-')[0]
    
    if xx in valid_xx:
        file_path = os.path.join(audio_dir, file)
        mfcc_features = extract_features(file_path, n_mfcc=13)
        features.append(mfcc_features)
        file_names.append(file)

if not features:
    print("No hay archivos que coincidan con los valores válidos de 'xx'.")
else:
    # Convertir a numpy array y normalizar
    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Parámetros que no varían
    clusters_list = 2
    init_params_list = "random"
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    n_init_list = 10
    tol_list = 1e-5
    reg_covar = 1e-4 

    # Etiquetas reales
    xx_values = [file.split('-')[0] for file in file_names]
    xx_labels = sorted(set(xx_values))

    # Lista de valores para max_iter
    # Para almacenar los accuracies
    accuracies = []

    for iter in covariance_types:

        # Aplicar GMM
        gmm = GaussianMixture(
            n_components=clusters_list,
            covariance_type=iter,
            n_init=n_init_list,
            init_params=init_params_list,
            reg_covar=reg_covar,
            tol=tol_list,
            max_iter=500,
            random_state=42
        )
        clusters = gmm.fit_predict(features)

        for i, file in enumerate(file_names):
            xx = file.split('-')[0]
            if xx.startswith(valid_xx[3]) or xx.startswith(valid_xx[6]):
                clusters[i] = 1 - clusters[i]

        # Crear matriz de confusión
        confusion_matrix = np.zeros((len(xx_labels), clusters_list), dtype=int)

        for xx, cluster in zip(xx_values, clusters):
            xx_index = xx_labels.index(xx)
            confusion_matrix[xx_index, cluster] += 1

        # Grupos de emociones
        xx_groups = [['01', '02', '03', '08'], ['04', '05', '06', '07']]  # Agrupación por categorías

        # Crear matriz de confusión para los grupos
        group_confusion_matrix = np.zeros((len(xx_groups), clusters_list), dtype=int)

        # Mapear xx_values a sus respectivos grupos
        for xx, cluster in zip(xx_values, clusters):
            for group_index, group in enumerate(xx_groups):
                if xx in group:
                    group_confusion_matrix[group_index, cluster] += 1

        # Encontrar el mejor cluster para cada grupo
        best_clusters = []
        for group_index in range(len(xx_groups)):
            best_cluster = np.argmax(group_confusion_matrix[group_index, :])
            best_clusters.append(best_cluster)

        # Calcular el accuracy basado en los mejores clusters para cada grupo
        correct_predictions = 0
        total_samples = 0

        for group_index, best_cluster in enumerate(best_clusters):
            correct_predictions += group_confusion_matrix[group_index, best_cluster]
            total_samples += np.sum(group_confusion_matrix[group_index, :])

        accuracy = correct_predictions / total_samples
        accuracies.append(accuracy)


    # Gráfico de barras para comparar los accuracies
    plt.figure(figsize=(10, 6))
    plt.bar([str(x) for x in covariance_types], accuracies, color='#3B8CD8')
    plt.xlabel('Valores de max_iter')
    plt.ylabel('Accuracy')
    plt.title('Comparación de Accuracy para diferentes valores de max_iter')
    plt.tight_layout()
    plt.show()
