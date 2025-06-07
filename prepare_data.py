# prepare_data.py
import librosa
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "data/genres"
features, labels = [], []

for label, genre in enumerate(os.listdir(DATA_DIR)):
    genre_path = os.path.join(DATA_DIR, genre)
    for filename in tqdm(os.listdir(genre_path), desc=genre):
        file_path = os.path.join(genre_path, filename)
        try:
            y, sr = librosa.load(file_path, duration=30)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = mel_db[:128, :128]  # Ensure fixed shape
            if mel_db.shape == (128, 128):
                features.append(mel_db)
                labels.append(label)
        except:
            continue

np.save("data/features.npy", np.array(features))
np.save("data/labels.npy", np.array(labels))
