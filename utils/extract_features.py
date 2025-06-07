import librosa
import numpy as np

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:128, :128]
    return mel_db if mel_db.shape == (128, 128) else None

