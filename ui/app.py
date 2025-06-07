import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from utils.extract_features import extract_mel_spectrogram

# Load model and labels
try:
    model = load_model("model/genre_cnn_model.h5")
    genre_labels = np.load("model/label_classes.npy", allow_pickle=True)  # Load string labels
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    model = None
    genre_labels = []

def main():
    st.title("🎵 Music Genre Prediction App")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        if model is None:
            st.error("Model is not loaded.")
            return

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file.flush()
            audio_path = tmp_file.name

        try:
            features = extract_mel_spectrogram(audio_path)
            features = features / 255.0  # Normalize like training
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)

            prediction = model.predict(features)
            predicted_index = np.argmax(prediction, axis=1)[0]

            # Ensure predicted_index is valid index for genre_labels
            if len(genre_labels) > predicted_index:
                predicted_genre = genre_labels[predicted_index]
            else:
                predicted_genre = f"Unknown genre index {predicted_index}"

            # st.subheader("🎧 Prediction Probabilities:")
            # for i, prob in enumerate(prediction[0]):
            #     label = genre_labels[i] if i < len(genre_labels) else f"Index {i}"
            #     st.write(f"{label}: **{prob:.3f}**")

            st.success(f"🎶 Predicted genre: **{predicted_genre}**")

        except Exception as e:
            st.error(f"Error processing audio file: {e}")

if __name__ == "__main__":
    main()
