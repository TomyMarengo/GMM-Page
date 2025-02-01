import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Escalar cada característica (cada fila) por separado
    scaler = StandardScaler()
    for i in range(features.shape[1]):  # Iterar sobre las características
        features[:, i] = scaler.fit_transform(features[:, i].reshape(-1, 1)).flatten()

    n_clusters = 3
    # Aplicar Gaussian Mixture Model con 3 clusters
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        n_init=10,
        init_params="random",
        reg_covar=1e-8,
        tol=1e-5,
        max_iter=1000,
        random_state=42
    )
    clusters = gmm.fit_predict(features)

    # Crear lista de los primeros dígitos del nombre del archivo
    xx_values = [file.split('-')[0] for file in file_names]

    # Crear la matriz de confusión
    confusion_matrix = np.zeros((len(set(xx_values)), n_clusters), dtype=int)

    # Rellenar la matriz de confusión
    for xx, cluster in zip(xx_values, clusters):
        xx_index = list(set(xx_values)).index(xx)
        confusion_matrix[xx_index, cluster] += 1

    y_label = ["Calmo", "Feliz", "Enojado"]
    # Crear el heatmap
    xx_labels = sorted(set(xx_values))
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f'Cluster {i}' for i in range(n_clusters)], 
                yticklabels=y_label, cbar=True)

    # Personalizar el gráfico
    plt.xlabel('Clusters')
    plt.ylabel('xx (primeros dígitos del archivo)')
    plt.title('Matriz de Confusión: Cluster vs xx')
    plt.show()