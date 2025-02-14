import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib 

# Load dataset
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    X = np.array(data["mfcc"])  # Fitur MFCC
    y = np.array(data["labels"])  # Label (0: kucing, 1: anjing)
    
    return X, y

# Load dataset
json_path = "dataset/mfcc_features.json"
X, y = load_data(json_path)

# Normalisasi fitur menggunakan StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ubah shape agar sesuai input Conv1D (samples, timesteps, features)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Bangun Model CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fungsi untuk melatih model
def train_model():
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
    
    # Simpan model setelah training
    model.save("src/saved_model.keras")
    
    return history
