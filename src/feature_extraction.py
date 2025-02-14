import os
import numpy as np
import librosa
import json

# Parameters
sr = 22050  # Sample rate
duration = 2.5  # Audio duration in seconds
n_mfcc = 40  # Number of MFCCs
n_fft = 2048
hop_length = 512

def extract_features(file_path):
    """Ekstraksi fitur MFCC dari file audio."""
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    target_length = int(sr * duration)  # Hitung panjang target
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))  # Padding jika lebih pendek
    else:
        y = y[:target_length]  # Potong jika lebih panjang
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = np.mean(mfccs.T, axis=0)  # Ambil rata-rata MFCC
    return mfccs

def process_dataset(input_dir, output_json):
    """Proses semua file dalam dataset dan simpan fitur dalam JSON."""
    data = {"mfcc": [], "labels": [], "files": []}
    labels_map = {"cat": 0, "dog": 1}  # Label encoding

    for label in ["cat", "dog"]:
        class_dir = os.path.join(input_dir, label)
        if not os.path.exists(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                mfcc_features = extract_features(file_path)

                data["mfcc"].append(mfcc_features.tolist())  # Simpan dalam bentuk list
                data["labels"].append(labels_map[label])  # Simpan label numerik
                data["files"].append(file_name)  # Simpan nama file
    
    # Simpan ke JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved features to {output_json}")

if __name__ == "__main__":
    dataset_dir = "dataset/cleaned_audio"
    output_json = "dataset/mfcc_features.json"
    
    process_dataset(dataset_dir, output_json)
