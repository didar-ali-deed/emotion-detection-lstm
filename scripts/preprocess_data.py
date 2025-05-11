# scripts/preprocess_data.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
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

def preprocess_data(data_dir="../data", max_words=5000, max_len=100):
    """
    Preprocess the Emotion dataset: clean, tokenize, pad, and encode labels.
    """
    # Load CSV files
    train_df = pd.read_csv(os.path.join(data_dir, "emotion_train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "emotion_test.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "emotion_validation.csv"))

    # Clean text
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    val_df['text'] = val_df['text'].apply(clean_text)

    # Initialize and fit tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['text'])

    # Convert text to sequences
    X_train = tokenizer.texts_to_sequences(train_df['text'])
    X_test = tokenizer.texts_to_sequences(test_df['text'])
    X_val = tokenizer.texts_to_sequences(val_df['text'])

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
    X_val = pad_sequences(X_val, maxlen=max_len, padding='post', truncating='post')

    # Encode labels (one-hot encoding)
    y_train = pd.get_dummies(train_df['label']).values
    y_test = pd.get_dummies(test_df['label']).values
    y_val = pd.get_dummies(val_df['label']).values

    # Save preprocessed data
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)

    # Save tokenizer
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    return X_train, X_test, X_val, y_train, y_test, y_val, tokenizer

if __name__ == "__main__":
    X_train, X_test, X_val, y_train, y_test, y_val, tokenizer = preprocess_data()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Sample tokenized sequence:", X_train[0])
    print("Sample label (one-hot):", y_train[0])
    print("Tokenizer saved to models/tokenizer.pkl")