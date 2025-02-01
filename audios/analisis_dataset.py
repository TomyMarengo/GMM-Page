import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Configuración de seaborn
sns.set(style="darkgrid")

# Carpeta de audios
audio_folder = "./audios"

# Etiquetas de emociones
emotion_labels = {
    "01": "Neutral",
    "02": "Calmado",
    "03": "Feliz",
    "04": "Triste",
    "05": "Enojado",
    "06": "Miedo",
    "07": "Disgusto",
    "08": "Sorpresa"
}

# Diccionarios para almacenar métricas por prefijo
energies = defaultdict(list)
durations = defaultdict(list)
zcrs = defaultdict(list)  # Tasa de Cruce por Cero
rms = defaultdict(list)  # Root Mean Square Energy
mfccs_dict = defaultdict(list)  # MFCCs Promedios

# Leer todos los audios de la carpeta
for file in sorted(os.listdir(audio_folder)):
    if file.endswith(".wav"):
        # Extraer el prefijo del nombre
        prefix = file.split('-')[0]
        file_path = os.path.join(audio_folder, file)
        y, sr = librosa.load(file_path, sr=None)

        # Calcular métricas
        duration = librosa.get_duration(y=y, sr=sr)
        energy = np.sum(y**2) / len(y)  # Energía normalizada
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # Tasa de Cruce por Cero
        rms_value = np.mean(librosa.feature.rms(y=y))  # Root Mean Square Energy
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

        # Guardar en el grupo correspondiente
        durations[prefix].append(duration)
        energies[prefix].append(energy)
        zcrs[prefix].append(zcr)
        rms[prefix].append(rms_value)
        mfccs_dict[prefix].append(mfccs)

# Calcular promedios y desviación estándar por prefijo
prefixes = sorted(emotion_labels.keys())
avg_durations = [np.mean(durations[p]) for p in prefixes]
std_durations = [np.std(durations[p]) for p in prefixes]
avg_energies = [np.mean(energies[p]) for p in prefixes]
std_energies = [np.std(energies[p]) for p in prefixes]
avg_zcrs = [np.mean(zcrs[p]) for p in prefixes]
std_zcrs = [np.std(zcrs[p]) for p in prefixes]
avg_rms = [np.mean(rms[p]) for p in prefixes]
std_rms = [np.std(rms[p]) for p in prefixes]

# Crear un DataFrame para estadísticas generales
data_summary = {
    "Emoción": [emotion_labels[p] for p in prefixes],
    "Duración Promedio (s)": avg_durations,
    "Desviación Duración": std_durations,
    "Energía Promedio": avg_energies,
    "Desviación Energía": std_energies,
    "Tasa de Cruce por Cero Promedio": avg_zcrs,
    "Desviación ZCR": std_zcrs,
    "Energía RMS Promedio": avg_rms,
    "Desviación RMS": std_rms
}
df_summary = pd.DataFrame(data_summary)

# Graficar duración promedio con desviación estándar
# Graficar duración promedio con desviación estándar
plt.figure(figsize=(10, 5))
plt.bar(df_summary["Emoción"], df_summary["Duración Promedio (s)"], 
        yerr=df_summary["Desviación Duración"], color="#3B8CD8", capsize=5)
plt.xlabel("Emociones")
plt.ylabel("Duración Promedio (s)")
plt.xticks(rotation=45)
plt.show()

# Graficar energía promedio con desviación estándar
plt.figure(figsize=(10, 5))
plt.bar(df_summary["Emoción"], df_summary["Energía Promedio"], 
        yerr=df_summary["Desviación Energía"], color="#3B8CD8", capsize=5)
plt.xlabel("Emociones")
plt.ylabel("Energía Promedio")
plt.xticks(rotation=45)
plt.show()

# Graficar tasa de cruce por cero promedio con desviación estándar
plt.figure(figsize=(10, 5))
plt.bar(df_summary["Emoción"], df_summary["Tasa de Cruce por Cero Promedio"], 
        yerr=df_summary["Desviación ZCR"], color="#3B8CD8", capsize=5)
plt.xlabel("Emociones")
plt.ylabel("Tasa de Cruce por Cero Promedio")
plt.xticks(rotation=45)
plt.show()
