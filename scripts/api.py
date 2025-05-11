# scripts/api.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean text by lowercasing, removing punctuation, and stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Load the saved model and tokenizer.
    """
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Load model and tokenizer at startup
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(models_dir, 'emotion_model.keras')
tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion for input text.
    Expects JSON: {"text": "input text"}
    Returns JSON: {"emotion": "predicted_emotion", "confidence": float}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in JSON'}), 400

        text = data['text']
        cleaned_text = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')

        prediction = model.predict(padded_sequence, verbose=0)
        emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
        pred_label = emotion_labels[np.argmax(prediction)]
        confidence = float(prediction[0].max())

        return jsonify({'emotion': pred_label, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)