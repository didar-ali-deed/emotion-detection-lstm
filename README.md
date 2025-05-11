# Emotion Detection from Text Using Deep Learning (LSTM)

This project uses a Bidirectional LSTM model to classify emotions (e.g., joy, sadness, anger) from text, leveraging the Hugging Face Emotion dataset. Built with TensorFlow/Keras, it showcases NLP techniques like tokenization, embeddings, and RNNs.

## Project Structure
```
data/        → Datasets (Emotion dataset from Hugging Face)
models/      → Trained models and tokenizer
notebooks/   → Jupyter notebooks for analysis
scripts/     → Python scripts for preprocessing, training, and deployment
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/didar-ali-deed/emotion-detection-lstm.git
   ```
2. Create a new conda environment:
   ```bash
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```bash
   conda activate emotion-lstm
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow and Progress

### Step 1: Dataset Selection
- Loaded and saved the Hugging Face Emotion dataset using `scripts/load_dataset.py`.
- Dataset exploration prints the first 5 rows, data types, and emotion distribution.
- The dataset is split into train, test, and validation sets and saved as CSV files in `data/`.

### Step 2: Data Preprocessing
Implemented in `scripts/preprocess_data.py`:
- **Text Cleaning:** Lowercases text, removes punctuation, and optionally removes stopwords.
- **Tokenization:** Builds a vocabulary of up to 5,000 words. Unknown words are mapped to `<OOV>`.
- **Sequence Padding:** Pads/truncates sequences to a maximum length of 100.
- **Label Encoding:** Converts labels to one-hot vectors for classification.
- **Saving:** Saves preprocessed arrays as `.npy` files for training.

### Step 3: Model Building
- Created a Bidirectional LSTM model with embedding and dropout layers in `scripts/build_model.py`.
- Saved model summary to `models/model_summary.txt`.

### Step 4: Model Training and Evaluation
- Trained the model using `scripts/train_evaluate.py`.
- Evaluated performance with accuracy, precision, recall, F1-score, and confusion matrix.
- Saved the trained model to `models/emotion_model.h5` and confusion matrix visualization to `models/confusion_matrix.png`.

### Step 5: Model Saving and Verification
- Saved the final trained model in `.keras` format using `scripts/save_verify_model.py`.
- Saved tokenizer as `models/tokenizer.pkl` for deployment.

### Step 6: Inference
- Developed an inference script (`scripts/inference.py`) to predict emotions from new text inputs using the saved model and tokenizer.

### Step 7: Flask API Deployment
- Created a Flask API (`scripts/api.py`) to serve the emotion detection model.
- Accepts text input via POST requests at:
  ```
  http://localhost:5000/predict
  ```

## Expected Outcome
- A robust emotion detection system based on LSTM.
- End-to-end pipeline from dataset to production-ready API.
- Code and model saved and committed to GitHub:
  [https://github.com/didar-ali-deed/emotion-detection-lstm](https://github.com/didar-ali-deed/emotion-detection-lstm)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.