import os
import librosa
import soundfile as sf

def remove_noise(audio):
    """Menghilangkan noise dari audio menggunakan pre-emphasis."""
    return librosa.effects.preemphasis(audio)

def process_audio_files(input_dir, output_dir):
    """Membersihkan semua file .wav dalam folder tertentu."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_dir, file_name)
            
            # Load audio
            audio, sr = librosa.load(file_path, sr=None)
            
            # Hilangkan noise
            audio_denoised = remove_noise(audio)
            
            # Simpan kembali audio yang sudah dibersihkan
            output_path = os.path.join(output_dir, file_name)
            sf.write(output_path, audio_denoised, sr)
            print(f"Processed: {file_name}")

if __name__ == "__main__":
    # Direktori dataset
    base_dir = "dataset/cats_dogs/train"
    cat_audio_dir = os.path.join(base_dir, "cat")
    dog_audio_dir = os.path.join(base_dir, "dog")

    # Direktori output
    cleaned_dir = "dataset/cleaned_audio"
    cat_output_dir = os.path.join(cleaned_dir, "cat")
    dog_output_dir = os.path.join(cleaned_dir, "dog")

    # Proses file suara kucing dan anjing
    process_audio_files(cat_audio_dir, cat_output_dir)
    process_audio_files(dog_audio_dir, dog_output_dir)
