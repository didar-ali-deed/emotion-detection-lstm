# scripts/inference.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

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

def predict_emotion(text, model, tokenizer, max_len=100):
    """
    Predict the emotion for a given text.
    Args:
        text (str): Input text.
        model: Loaded Keras model.
        tokenizer: Loaded tokenizer.
        max_len (int): Sequence length.
    Returns:
        str: Predicted emotion.
        float: Confidence score.
    """
    # Clean and preprocess text
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded_sequence, verbose=0)
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    pred_label = emotion_labels[np.argmax(prediction)]
    confidence = prediction[0].max()
    
    return pred_label, confidence

if __name__ == "__main__":
    # Load model and tokenizer
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_path = os.path.join(models_dir, 'emotion_model.keras')
    tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

    # Sample texts for testing
    sample_texts = [
        "I feel so sad and alone today",
        "This is the best day ever, I'm so happy!",
        "I'm really scared about the future",
        "I love spending time with my family",
        "I'm so angry at how things turned out"
    ]

    # Predict emotions
    print("\nEmotion Predictions:")
    for text in sample_texts:
        emotion, confidence = predict_emotion(text, model, tokenizer)
        print(f"Text: {text}")
        print(f"Predicted Emotion: {emotion}, Confidence: {confidence:.4f}\n")