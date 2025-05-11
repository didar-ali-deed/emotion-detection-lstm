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
-Exploration: explore_dataset prints the first 5 rows, data types, and emotion distribution to check for class imbalance or missing values.
-Saving: CSV files are saved in data/ for use in preprocessing (Step 2).
-Modularity: The script is placed in scripts/ to keep the project organized.

## License
MIT License