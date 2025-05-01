import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from src.utils import clean_text, summarize_text
from src.config import MODELS_DIR, DATA_PATH, CLASSIFICATION_REPORT_PATH, CONFUSION_MATRIX_PATH

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load pre-trained models
vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
ensemble = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.joblib"))

# Analyze and summarize the review
def analyze_review(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    sentiment_encoded = ensemble.predict(vec)[0]
    sentiment = "positive" if sentiment_encoded == 1 else "negative"  # Decode sentiment
    summary = summarize_text(text)  # Removed unnecessary arguments
    return sentiment, summary  # Return both sentiment and summary

# Confusion Matrix (for entire dataset)
def plot_confusion_matrix(y_true, y_pred):
    """Prints and plots the confusion matrix given true and predicted labels."""
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    disp.ax_.set_title("Confusion Matrix")

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    df = df[df['Score'].apply(lambda x: str(x).isdigit())]
    df["Score"] = df["Score"].astype(int)
    df = df[df["Score"] != 3].copy()
    df["sentiment"] = df["Score"].apply(lambda s: "positive" if s > 3 else "negative")
    df["Text_clean"] = df["Text"].apply(clean_text)

    # Use pre-trained vectorizer and model
    X_vec = vectorizer.transform(df["Text_clean"])
    y = df["sentiment"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Predict using the pre-trained model
    y_pred = ensemble.predict(X_vec)

    # Decode predicted labels to match the true labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Print and plot confusion matrix
    plot_confusion_matrix(y, y_pred_decoded)
