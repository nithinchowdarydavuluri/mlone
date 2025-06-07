import numpy as np

# Replace this list with your actual genre names
genre_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Save as numpy array of strings
np.save("model/label_classes.npy", np.array(genre_names))
print("Genre labels saved successfully.")
