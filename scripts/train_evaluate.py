# scripts/train_evaluate.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def build_model(vocab_size=5000, max_len=100, num_classes=6):
    """
    Build a Bidirectional LSTM model.
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
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    return X_train, X_test, X_val, y_train, y_test, y_val

def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Plot and save the confusion matrix.
    """
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Load data
    X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data()

    # Build model
    model = build_model()

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute confusion matrix and classification report
    emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
    print("\nClassification Report:")
    print(classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        target_names=emotion_labels
    ))

    # Plot and save confusion matrix
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, emotion_labels, os.path.join(models_dir, 'confusion_matrix.png'))

    # Save the trained model
    model.save(os.path.join(models_dir, 'emotion_model.h5'))

    print(f"Model saved to {os.path.join(models_dir, 'emotion_model.h5')}")
    print(f"Confusion matrix saved to {os.path.join(models_dir, 'confusion_matrix.png')}")