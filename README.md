# Emotion Detection from Text Using Deep Learning (LSTM)

This project uses a Bidirectional LSTM model to classify emotions (e.g., joy, sadness, anger) from text, leveraging the Hugging Face Emotion dataset. Built with TensorFlow/Keras, it showcases NLP techniques like tokenization, embeddings, and RNNs.

## Project Structure
- `data/`: Datasets (Emotion dataset from Hugging Face).
- `models/`: Trained models.
- `notebooks/`: Jupyter notebooks for analysis.
- `scripts/`: Python scripts for preprocessing, training, and evaluation.

## Setup
1. Clone the repository: `git clone https://github.com/didar-ali-deedy/emotion-detection-lstm.git`
2. Create an Anaconda environment: `conda env create -f environment.yml`
3. Activate the environment: `conda activate emotion-lstm`
4. Install dependencies: `pip install -r requirements.txt`

## Next Steps
- Load and preprocess the Emotion dataset.
- Train the LSTM model.
- Evaluate and deploy (optional).


## Progress
- **Step 1: Dataset Selection**: Loaded and saved the Hugging Face Emotion dataset using `scripts/load_dataset.py`.
**Explanation of Code**
- Loading: load_dataset("emotion") fetches the dataset from Hugging Face’s servers.
- Conversion: The dataset is split into train, test, and validation sets, converted to Pandas DataFrames for easy handling.
- Exploration: explore_dataset prints the first 5 rows, data types, and emotion distribution to check for class imbalance or missing values.
- Saving: CSV files are saved in data/ for use in preprocessing (Step 2).
- Modularity: The script is placed in scripts/ to keep the project organized.

## Progress
- **Step 2: Preprocess the Data**: Implemented text cleaning, tokenization, and sequence padding in `scripts/preprocess_data.py`.
- Cleaning: clean_text: Lowercases text, removes punctuation, and optionally removes stopwords. Stopwords removal is included but can be skipped for emotion detection (small words like “I” may carry emotional weight).
- Tokenization:
    - Tokenizer: Builds a vocabulary of up to max_words (5000) words, with <OOV> for unknown words.
    - fit_on_texts: Uses training data to create the vocabulary.
    - texts_to_sequences: Converts text to integer sequences.
- Padding:
     - pad_sequences: Pads sequences to max_len (100) with zeros or truncates them.
     - padding='post', truncating='post': Adds zeros or cuts off at the end.
- Label Encoding:
    - pd.get_dummies: Converts labels (0–5) to one-hot vectors (e.g., [1, 0, 0, 0, 0, 0] for sadness).
- Saving: Saves preprocessed data as .npy files for efficient loading in training.

## Progress
- **Step 3: Create the Model**: Built a Bidirectional LSTM model with Embedding and Dropout layers in `scripts/build_model.py`. Saved model summary to `models/model_summary.txt`.

## 🌟 Expected Outcome
- The Bidirectional LSTM model is built and compiled correctly.
- Model summary is saved in models/model_summary.txt.
- scripts/build_model.py is updated and committed.
- Changes are pushed to your GitHub repository (https://github.com/didar-ali-deed/emotion-detection-lstm).


## Progress
- **Step 4: Train and Evaluate**: Trained the Bidirectional LSTM model and evaluated it with accuracy, confusion matrix, and F1 score in `scripts/train_evaluate.py`. Saved trained model to `models/emotion_model.h5` and confusion matrix to `models/confusion_matrix.png`.

## Explanation of Code
- Model: Reuses the Bidirectional LSTM from Step 3 for consistency.
- Training:
    - model.fit: Trains for 10 epochs with batch size 32, using validation data to monitor performance.
    - Epochs and batch size are chosen for balance; adjust if needed (e.g., increase epochs if underfitting).
- Evaluation:
    - model.evaluate: Computes test accuracy.
    - classification_report: Provides precision, recall, and F1 score per class.
    - confusion_matrix: Shows correct and incorrect predictions per class.
- Visualization: plot_confusion_matrix creates a heatmap of the confusion matrix, saved as a PNG.

## Progress
- **Step 5: Save the Model**: Saved the trained Bidirectional LSTM model in .keras format and verified predictions in `scripts/save_verify_model.py`. Model saved to `models/emotion_model.keras`.



## License
MIT License