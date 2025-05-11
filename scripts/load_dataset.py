# scripts/load_dataset.py
import pandas as pd
from datasets import load_dataset
import os

def load_emotion_dataset(data_dir="../data"):
    """
    Load the Hugging Face Emotion dataset and save it as CSV files.
    Args:
        data_dir (str): Directory to save the dataset.
    Returns:
        train_df, test_df, validation_df: Pandas DataFrames for each split.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Load the dataset
    dataset = load_dataset("emotion")

    # Convert to Pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    validation_df = pd.DataFrame(dataset['validation'])

    # Save to CSV
    train_df.to_csv(os.path.join(data_dir, "emotion_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "emotion_test.csv"), index=False)
    validation_df.to_csv(os.path.join(data_dir, "emotion_validation.csv"), index=False)

    return train_df, test_df, validation_df

def explore_dataset(df, name="Dataset"):
    """
    Explore the dataset's structure and emotion distribution.
    Args:
        df: Pandas DataFrame.
        name: Name of the dataset split (e.g., Train, Test).
    """
    print(f"\n{name} Overview:")
    print(df.head())
    print(f"\n{name} Info:")
    print(df.info())
    print(f"\n{name} Emotion Distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    # Load the dataset
    train_df, test_df, validation_df = load_emotion_dataset()

    # Explore each split
    explore_dataset(train_df, "Training Data")
    explore_dataset(test_df, "Test Data")
    explore_dataset(validation_df, "Validation Data")