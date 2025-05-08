# mfcc_to_images.py

import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Chargement des MFCC ---
mfcc_data = np.load("mfcc_clean.npy")

# --- 2. Créer un dossier pour les images ---
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# --- 3. Générer les images ---
for i, mfcc in enumerate(mfcc_data):
    plt.figure(figsize=(2.24, 2.24), dpi=100)  # pour obtenir des images proches de 224x224
    plt.imshow(mfcc.reshape(1, -1), aspect='auto', cmap='viridis')  # MFCC sous forme d'image 2D
    plt.axis('off')  # pas d'axe

    # Sauvegarde
    filename = f"mfcc_{i:04d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

print(f"✅ {len(mfcc_data)} images MFCC générées dans '{output_dir}/'")
