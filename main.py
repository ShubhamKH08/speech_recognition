from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the saved model and mappings
model = load_model('speech_recognition/speech_to_text_model_.h5')

with open('speech_recognition/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open('speech_recognition/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

max_time_steps = 300  # Adjust as per your model training
decoder_input_data = np.load('speech_recognition/decoder_input_data.npy')

# Prediction function
def predict_audio_transcript(audio_path):
    y, _ = librosa.load(audio_path, sr=16000)
    spectrogram = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).T
    spectrogram_padded = np.pad(spectrogram, ((0, max_time_steps - spectrogram.shape[0]), (0, 0)), mode='constant')

    predicted_sequence = model.predict([spectrogram_padded[np.newaxis, ...], decoder_input_data[:1]])
    predicted_indices = np.argmax(predicted_sequence, axis=-1)
    predicted_words = [index_to_word.get(idx, '<UNK>') for idx in predicted_indices[0] if idx > 0]

    return " ".join(predicted_words)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling file upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save file temporarily
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    try:
        # Predict transcript
        transcript = predict_audio_transcript(file_path)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(file_path)  # Remove file after prediction

    return jsonify({'transcript': transcript})


import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

