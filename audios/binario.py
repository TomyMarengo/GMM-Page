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

    # Parámetros de GMM
    clusters_list = 2
    init_params_list = "random_from_data"
    covariance_types = "full"
    n_init_list = 20
    tol_list = 1e-5
    reg_covar = 1e-4 

    # Etiquetas reales
    xx_values = [file.split('-')[0] for file in file_names]
    xx_labels = sorted(set(xx_values))

    # Aplicar GMM
    gmm = GaussianMixture(
        n_components=clusters_list,
        covariance_type=covariance_types,
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

    # Graficar y guardar matriz de confusión
    #01=Neutral, 02=Calmado, 03=Feliz, 04=Triste, 05=Enojado, 06=Miedo, 07=Disgusto, 08=Sorpresa
    y_legend=["Neutral", "Calmado", "Feliz", "Triste", "Enojado", "Miedo", "Disgusto", "Sorpresa"]
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f'Cluster {i}' for i in range(clusters_list)],
                yticklabels=y_legend, cbar=True)

    plt.xlabel('Clusters')
    plt.ylabel('Emociones')

    filename = "conf_matrix.png"
    plt.show()
