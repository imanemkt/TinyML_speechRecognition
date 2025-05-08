<<<<<<< HEAD
import pandas as pd
import os
import librosa
import numpy as np
import noisereduce as nr

### --- 1. Charger et sous-échantillonner (1%) ---
df = pd.read_csv("dataset_combined.csv")
df_sample = df.sample(frac=0.01, random_state=42)
df_sample.to_csv("clean1%.csv", index=False)
print(f"✅ Sample créé avec {len(df_sample)} lignes")

### --- 2. Nettoyage des données ---
def is_valid_audio(path):
    try:
        duration = librosa.get_duration(path=path)
        return 2.0 <= duration <= 6.5
    except Exception:
        return False

df_sample.dropna(subset=["audio_path", "sentence"], inplace=True)
df_sample.drop_duplicates(subset="sentence", inplace=True)
df_sample.drop_duplicates(subset="audio_path", inplace=True)
df_sample = df_sample[df_sample["audio_path"].apply(os.path.exists)]
df_sample = df_sample[df_sample["audio_path"].apply(is_valid_audio)]
df_sample = df_sample[df_sample["sentence"].str.split().str.len().between(3, 20)]
df_sample.to_csv("clean1%.csv", index=False)
print(f"✅ Nettoyage terminé : {len(df_sample)} lignes dans clean1%.csv")

### --- 3. Extraction MFCC avec réduction de bruit ---
def extract_mfcc_denoised(audio_path, n_mfcc=13, sr=16000):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"⚠️ Erreur avec {audio_path}: {e}")
        return None

mfcc_features = []
labels = []

for idx, row in df_sample.iterrows():
    mfcc = extract_mfcc_denoised(row["audio_path"])
    if mfcc is not None:
        mfcc_features.append(mfcc)
        labels.append(row["sentence"])

X = np.array(mfcc_features)
y = np.array(labels)

# Sauvegarde MFCCs en .npy
np.save("mfcc_clean.npy", X)

# Sauvegarde MFCCs en .txt
with open("mfcc_clean.txt", "w", encoding="utf-8") as f:
    for vecteur in X:
        f.write(" ".join(map(str, vecteur)) + "\n")

# Sauvegarde des labels en .txt
with open("labels_clean.txt", "w", encoding="utf-8") as f:
    for phrase in y:
        f.write(phrase + "\n")

print("✅ Extraction MFCC avec réduction de bruit terminée.")
print(f"✅ {X.shape[0]} fichiers audio traités.")
=======
import pandas as pd
import os
import librosa
import numpy as np

### --- 1. Charger et sous-échantillonner (1%) ---
df = pd.read_csv("dataset_combined.csv")

# Prendre 1% aléatoire
df_sample = df.sample(frac=0.01, random_state=42)
df_sample.to_csv("clean1%.csv", index=False)

print(f"✅ Sample créé avec {len(df_sample)} lignes")

### --- 2. Nettoyage ---
def is_valid_audio(path):
    try:
        duration = librosa.get_duration(path=path)
        return 2.0 <= duration <= 6.5
    except Exception:
        return False

df_sample.dropna(subset=["audio_path", "sentence"], inplace=True)
df_sample.drop_duplicates(subset="sentence", inplace=True)
df_sample.drop_duplicates(subset="audio_path", inplace=True)
df_sample = df_sample[df_sample["audio_path"].apply(os.path.exists)]
df_sample = df_sample[df_sample["audio_path"].apply(is_valid_audio)]
df_sample = df_sample[df_sample["sentence"].str.split().str.len().between(3, 20)]

df_sample.to_csv("clean1%.csv", index=False)
print(f"✅ Nettoyage terminé : {len(df_sample)} lignes dans clean1%.csv")

### --- 3. Extraction MFCC ---
def extract_mfcc(audio_path, n_mfcc=13, sr=16000):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"⚠️ Erreur avec {audio_path}: {e}")
        return None

mfcc_features = []
labels = []

for idx, row in df_sample.iterrows():
    mfcc = extract_mfcc(row["audio_path"])
    if mfcc is not None:
        mfcc_features.append(mfcc)
        labels.append(row["sentence"])

X = np.array(mfcc_features)
y = np.array(labels)

# ✅ Sauvegarde des MFCC dans .npy (optionnel)
np.save("mfcc_clean1percent.npy", X)

# ✅ Sauvegarde des MFCC dans .txt
with open("mfcc_clean1percent.txt", "w", encoding="utf-8") as f:
    for vecteur in X:
        f.write(" ".join(map(str, vecteur)) + "\n")

# ✅ Sauvegarde des labels dans .txt
with open("labels_clean1percent.txt", "w", encoding="utf-8") as f:
    for phrase in y:
        f.write(phrase + "\n")

print("✅ Extraction MFCC terminée.")
print(f"✅ {X.shape[0]} fichiers audio traités.")
>>>>>>> e207a66205cc9c5b3710d64c13826f7f8cb9a8f5
