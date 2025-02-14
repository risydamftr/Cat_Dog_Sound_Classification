import os
import librosa
from src.preprocess import process_audio_files
from src.feature_extraction import process_dataset
import src.train as train
import matplotlib.pyplot as plt

#Load Dataset
# Path ke dataset
cat_audio_dir = "dataset/cats_dogs/train/cat/"
dog_audio_dir = "dataset/cats_dogs/train/dog/"

# Take up the .wav file
cat_files = [os.path.join(cat_audio_dir, f) for f in os.listdir(cat_audio_dir) if f.endswith(".wav")]
dog_files = [os.path.join(dog_audio_dir, f) for f in os.listdir(dog_audio_dir) if f.endswith(".wav")]

print(f"Total file kucing: {len(cat_files)}")
print(f"Total file anjing: {len(dog_files)}")

# Reading file
sample_file = cat_files[0]
y, sr = librosa.load(sample_file, sr=None)

print(f"Sample Rate: {sr}, Durasi: {len(y)/sr:.2f} detik")
print(f"Waveform Shape: {y.shape}")

#Preprocessing (Cleaning audio)
# Dataset directory
base_dir = "dataset/cats_dogs/train"
cat_audio_dir = os.path.join(base_dir, "cat")
dog_audio_dir = os.path.join(base_dir, "dog")

# Output directory
cleaned_dir = "dataset/cleaned_audio"
cat_output_dir = os.path.join(cleaned_dir, "cat")
dog_output_dir = os.path.join(cleaned_dir, "dog")

# Running the cleaning audio function
process_audio_files(cat_audio_dir, cat_output_dir)
process_audio_files(dog_audio_dir, dog_output_dir)

#Feature Extraction MFCC
output_json = "dataset/mfcc_features.json"
process_dataset(cleaned_dir, output_json)

# Training the model
history = train.train_model()

# Plot the training result
def plot_training(history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.savefig("src/training_plot.png")
    plt.show()

# Show diagram
plot_training(history)
