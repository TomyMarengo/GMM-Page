import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Directorio donde están los archivos de audio
audio_dir = "./audios"

# Lista de valores 'xx' que deseas incluir en el análisis
valid_xx = ['02', '03', '05']
autores = ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21', '23']

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extracción de MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Extracción de Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Energía (Root Mean Square)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    
    # Chroma (Tonalidad)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Concatenar características
    return np.hstack([np.mean(mfccs, axis=1), np.mean(zcr, axis=1), energy, np.mean(chroma, axis=1)])

# Recorrer los archivos de audio y extraer características
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
features = []
file_names = []

for file in audio_files:
    xx = file.split('-')[0]
    autor = file.split('-')[2].split('.')[0]
    
    if xx in valid_xx and autor in autores:
        file_path = os.path.join(audio_dir, file)
        mfcc_features = extract_features(file_path, n_mfcc=13)
        features.append(mfcc_features)
        file_names.append(file)

# Verificar si hay archivos que coinciden con los valores válidos
if len(features) == 0:
    print("No hay archivos que coincidan con los valores válidos de 'xx'.")
else:
    # Convertir a numpy array
    features = np.array(features)

    # Escalar características
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    n_clusters = 3
    n_init_values = [1e-9,1e-8, 1e-6, 1e-4, 1e-2, 1]
    accuracy_results = []

    for n_init in n_init_values:
        # Aplicar Gaussian Mixture Model con init_params="random"
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            n_init=10,
            init_params="random",
            reg_covar=n_init,
            tol=1e-5,
            max_iter=1000,
            random_state=42
        )
        clusters = gmm.fit_predict(features)

        # Obtener las clases reales (primeros dos dígitos del archivo)
        xx_values = [file.split('-')[0] for file in file_names]
        unique_xx = sorted(set(xx_values))  # Clases reales únicas

        # Crear matriz de confusión
        confusion_matrix = np.zeros((len(unique_xx), n_clusters), dtype=int)

        for xx, cluster in zip(xx_values, clusters):
            xx_index = unique_xx.index(xx)
            confusion_matrix[xx_index, cluster] += 1

        # Asignar clusters a clases según la mayor frecuencia
        cluster_to_class = {}
        assigned_clusters = set()
        for xx_index, row in enumerate(confusion_matrix):
            max_cluster = np.argmax(row)  # Cluster con más elementos en la clase
            if max_cluster not in assigned_clusters:
                cluster_to_class[max_cluster] = xx_index
                assigned_clusters.add(max_cluster)

        # Calcular accuracy
        correct_predictions = sum(
            confusion_matrix[xx_index, cluster] for cluster, xx_index in cluster_to_class.items()
        )
        total_samples = np.sum(confusion_matrix)
        accuracy = correct_predictions / total_samples

        accuracy_results.append(accuracy)

    # Graficar los resultados
    plt.figure(figsize=(8, 5))
    plt.plot(n_init_values, accuracy_results, marker='o', linestyle='-', color='#3B8CD8')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
