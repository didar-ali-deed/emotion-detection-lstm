# scripts/build_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import os

def build_model(vocab_size=5000, max_len=100, num_classes=6):
    """
    Build a Bidirectional LSTM model for emotion detection.
    Args:
        vocab_size (int): Vocabulary size (from tokenizer).
        max_len (int): Sequence length (from preprocessing).
        num_classes (int): Number of emotion classes.
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    return model

def load_preprocessed_data(data_dir="../data"):
    """
    Load preprocessed data from .npy files.
    Returns:
        X_train, X_test, X_val, y_train, y_test, y_val: Preprocessed data.
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data()

    # Build the model
    model = build_model()

    # Ensure models directory exists
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    # Save model summary to a text file
    summary_path = os.path.join(models_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    print(f"Model built and summary saved in {summary_path}")