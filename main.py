from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import tensorflow as tf

app = Flask(__name__)

# Load the model and required components
model = load_model('/speech_to_text_model_.h5')

with open('/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open('/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

max_time_steps = 300
decoder_input_data = np.load('/decoder_input_data.npy')

# Define the prediction function
def predict_audio_transcript(audio_path):
    y, _ = librosa.load(audio_path, sr=16000)
    spectrogram = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).T
    spectrogram_padded = np.pad(spectrogram, ((0, max_time_steps - spectrogram.shape[0]), (0, 0)), mode='constant')

    predicted_sequence = model.predict([spectrogram_padded[np.newaxis, ...], decoder_input_data[:1]])
    predicted_indices = np.argmax(predicted_sequence, axis=-1)
    predicted_words = [index_to_word.get(idx, '<UNK>') for idx in predicted_indices[0] if idx > 0]

    return " ".join(predicted_words)

# Create a Flask route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    audio_path = '/content/drive/MyDrive/DL/Libri/train/audio/19-198-0009.flac'
    audio_file.save(audio_path)

    # Get the predicted transcript
    try:
        predicted_transcript = predict_audio_transcript(audio_path)
        return jsonify({'predicted_transcript': predicted_transcript})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
