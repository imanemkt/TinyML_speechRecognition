<<<<<<< HEAD
# prepare_training_data.py

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# --- 1. Chargement des données extraites ---
X = np.load("mfcc_clean.npy")

with open("labels_clean.txt", "r", encoding="utf-8") as f:
    y = [line.strip() for line in f]

# --- 2. Encodage des labels (phrases -> entiers) ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- 3. Normalisation des MFCC ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Séparation en ensembles d'entraînement et de test ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# --- 5. Sauvegarde des ensembles ---
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# --- 6. Sauvegarde des objets utiles pour prédiction ---
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# --- 7. Affichage ---
print(f"✅ Données préparées pour l'apprentissage :")
print(f"   X_train : {X_train.shape}, y_train : {y_train.shape}")
print(f"   X_test  : {X_test.shape}, y_test  : {y_test.shape}")
=======
# encoder.py

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- 4. Préparation des données pour l'apprentissage ---

# Chargement des caractéristiques MFCC et des labels
X = np.load("mfcc_clean1percent.npy")

with open("labels_clean1percent.txt", "r", encoding="utf-8") as f:
    y = [line.strip() for line in f]

# Encodage des labels (phrases) en valeurs numériques
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisation des caractéristiques (MFCC)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Affichage des formes des ensembles pour vérifier la séparation
print(f"✅ Ensemble d'entraînement : X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"✅ Ensemble de test : X_test = {X_test.shape}, y_test = {y_test.shape}")
>>>>>>> e207a66205cc9c5b3710d64c13826f7f8cb9a8f5
