import os
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Directorio donde están los archivos de audio
audio_dir = "./audios"

# Lista de valores 'xx' válidos y su mapeo a emociones
valid_xx = ['01', '02', '03', '04', '05', '06', '07', '08']
xx_to_emotion = {
    '01': 'Neutral', '02': 'Calmado', '03': 'Feliz', '04': 'Triste',
    '05': 'Enojado', '06': 'Miedo', '07': 'Disgusto', '08': 'Sorpresa'
}

def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    return np.hstack([np.mean(mfccs, axis=1), np.mean(zcr, axis=1), energy, np.mean(chroma, axis=1)])

# Extraer características
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
features_dict = {}

for file in audio_files:
    xx = file.split('-')[0]
    if xx in valid_xx:
        file_path = os.path.join(audio_dir, file)
        feature_vector = extract_features(file_path)
        if xx not in features_dict:
            features_dict[xx] = []
        features_dict[xx].append(feature_vector)

# Calcular promedios de cada característica por 'xx'
averaged_features = {}
for xx, feature_list in features_dict.items():
    averaged_features[xx] = np.mean(feature_list, axis=0)

# Convertir a matriz para heatmap
xx_labels = sorted(averaged_features.keys())
feature_matrix = np.array([averaged_features[xx] for xx in xx_labels]).T

# Estandarizar cada fila (característica) por separado
scaler = StandardScaler()
for i in range(feature_matrix.shape[0]):
    feature_matrix[i, :] = scaler.fit_transform(feature_matrix[i, :].reshape(-1, 1)).flatten()

# Nombres de las características
feature_names = [f"MFCC_{i+1}" for i in range(13)] + ["ZCR", "Energy"] + [f"Chroma_{i+1}" for i in range(12)]

# Crear el heatmap con etiquetas descriptivas en el eje X
emotion_labels = [xx_to_emotion[xx] for xx in xx_labels]

plt.figure(figsize=(12, 8))
sns.heatmap(feature_matrix, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=emotion_labels, yticklabels=feature_names, cbar=True)
plt.xlabel("Emociones")
plt.ylabel("Características")
plt.title("Promedio de Características Estandarizadas por Emoción")
plt.show()
