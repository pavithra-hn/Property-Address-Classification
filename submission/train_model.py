import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# File paths
TRAIN_DATA_PATH = 'task_dataset - training_dataset.csv'
VAL_DATA_PATH = 'task_dataset - validation_dataset.csv'
MODEL_DIR = 'best_model'
APPROACH_FILE = 'approach.txt'

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded data from {path}: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {path}: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and digits (keep only alphabets and spaces for simplicity, 
    # though addresses might need numbers, let's keep it simple first or maybe keep numbers)
    # For addresses, numbers are important (e.g., Flat 101). Let's keep numbers.
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_and_evaluate():
    print("Loading data...")
    train_df = load_data(TRAIN_DATA_PATH)
    val_df = load_data(VAL_DATA_PATH)

    if train_df is None or val_df is None:
        return

    # Basic preprocessing
    print("Preprocessing data...")
    train_df['clean_text'] = train_df['property_address'].apply(preprocess_text)
    val_df['clean_text'] = val_df['property_address'].apply(preprocess_text)

    # Target variable
    X_train = train_df['clean_text']
    y_train = train_df['categories']
    X_val = val_df['clean_text']
    y_val = val_df['categories']

    # Pipeline: TF-IDF + Logistic Regression
    # Logistic Regression is a strong baseline for text classification
    print("Training model...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    print("Evaluating model...")
    y_pred = pipeline.predict(X_val)

    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)
    print("\nConfusion Matrix:\n")
    print(conf_matrix)

    # Save model and artifacts
    print(f"Saving artifacts to {MODEL_DIR}...")
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'property_classifier.pkl'))
    
    # Save metrics to a file
    with open(os.path.join(MODEL_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))

    print("Done.")

if __name__ == "__main__":
    train_and_evaluate()
