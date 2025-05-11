# scripts/preprocess_data.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean text by lowercasing, removing punctuation, and stopwords.
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords (optional, can be skipped for emotion detection)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def preprocess_data(data_dir="../data", max_words=5000, max_len=100):
    """
    Preprocess the Emotion dataset: clean, tokenize, pad, and encode labels.
    Args:
        data_dir (str): Directory with CSV files.
        max_words (int): Maximum vocabulary size.
        max_len (int): Maximum sequence length.
    Returns:
        X_train, X_test, X_val: Padded sequences.
        y_train, y_test, y_val: One-hot encoded labels.
        tokenizer: Fitted Keras Tokenizer.
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

    return X_train, X_test, X_val, y_train, y_test, y_val, tokenizer

if __name__ == "__main__":
    # Preprocess the data
    X_train, X_test, X_val, y_train, y_test, y_val, tokenizer = preprocess_data()

    # Print shapes to verify
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Sample tokenized sequence:", X_train[0])
    print("Sample label (one-hot):", y_train[0])