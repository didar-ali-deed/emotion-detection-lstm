# scripts/save_verify_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import os

def build_model(vocab_size=5000, max_len=100, num_classes=6):
    """
    Build a Bidirectional LSTM model.
    Args:
        vocab_size (int): Vocabulary size.
        max_len (int): Sequence length.
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    return X_train, X_test, X_val, y_train, y_test, y_val

def train_and_save_model(X_train, y_train, X_val, y_val, models_dir, max_words=5000, max_len=100):
    """
    Train the model and save it in .keras format.
    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        models_dir (str): Directory to save the model.
        max_words (int): Vocabulary size.
        max_len (int): Sequence length.
    Returns:
        model: Trained model.
    """
    model = build_model(vocab_size=max_words, max_len=max_len)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    
    # Save the model in .keras format
    model_path = os.path.join(models_dir, 'emotion_model.keras')
    model.save(model_path)
    
    return model

def load_and_verify_model(model_path, X_test, y_test, num_samples=5):
    """
    Load the saved model and verify with sample predictions.
    Args:
        model_path (str): Path to the saved model.
        X_test, y_test: Test data and labels.
        num_samples (int): Number of samples to predict.
    """
    model = load_model(model_path)
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    
    # Predict on test samples
    sample_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    sample_X = X_test[sample_indices]
    sample_y = y_test[sample_indices]
    predictions = model.predict(sample_X)
    
    print("\nSample Predictions:")
    for i in range(num_samples):
        true_label = emotion_labels[np.argmax(sample_y[i])]
        pred_label = emotion_labels[np.argmax(predictions[i])]
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label}, Confidence={predictions[i].max():.4f}")

    # Evaluate model accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nReloaded Model Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data()

    # Ensure models directory exists
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    # Train and save model
    model = train_and_save_model(X_train, y_train, X_val, y_val, models_dir)

    # Verify the saved model
    model_path = os.path.join(models_dir, 'emotion_model.keras')
    load_and_verify_model(model_path, X_test, y_test)

    print(f"Model saved to {model_path}")