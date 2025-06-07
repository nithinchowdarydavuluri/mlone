# train_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load features and labels
X = np.load("data/features.npy")
y_raw = np.load("data/labels.npy")

# Reshape features to (samples, 128, 128, 1)
if len(X.shape) == 3:
    X = X.reshape(-1, 128, 128, 1)

# Normalize
X = X / 255.0

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
y = to_categorical(y_encoded, num_classes=10)

# Save label classes for inference
os.makedirs("model", exist_ok=True)
np.save("model/label_classes.npy", le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded
)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=2
)

# Save model
model.save("model/genre_cnn_model.h5")
print("Model and labels saved.")


