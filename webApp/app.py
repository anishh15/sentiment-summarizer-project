# app.py – Streamlit UI for Sentiment + Summary Analysis

import streamlit as st
import joblib
import nltk
import pandas as pd
import os
from nltk.corpus import stopwords
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MODELS_DIR, VISUALS_DIR, CLASSIFICATION_REPORT_PATH, CONFUSION_MATRIX_PATH
from src.utils import clean_text, summarize_text

# Ensure NLTK stopwords corpus is downloaded before use
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
stop_words.discard("not")  # Keep "not" to preserve sentiment

# Replace hardcoded paths with config variables
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
REPORT_PATH = CLASSIFICATION_REPORT_PATH
CONF_MATRIX_IMG = CONFUSION_MATRIX_PATH

# Added error handling for model and vectorizer loading
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(CLASSIFIER_PATH)
except FileNotFoundError as e:
    st.error(f"Model or vectorizer file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Replace T5 summarizer with Hugging Face Inference API
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGING_FACE_API_KEY = os.getenv("HF_TOKEN")  # Load from environment variable

# Streamlit UI
st.set_page_config(page_title="Sentiment & Summary App", layout="centered")
st.title("📝 Amazon Review Analyzer")
st.write("Predict **sentiment** and generate a **summary** for Amazon reviews!")

menu = ["Single Review", "Batch Upload", "Model and Dataset"]
choice = st.sidebar.selectbox("Select Mode", menu)

if choice == "Single Review":
    review_input = st.text_area("Enter your review below:", height=200)

    if st.button("🔍 Analyze Review"):
        if review_input.strip():
            with st.spinner("Analyzing..."):
                cleaned = clean_text(review_input)
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                summary = summarize_text(review_input)

            # Decode prediction to string label before displaying
            sentiment = "positive" if prediction == 1 else "negative"
            st.success(f"**Predicted Sentiment:** {sentiment.capitalize()}")
            st.write("**🔢 Prediction Probabilities:**")
            st.write({"Negative": round(proba[0]*100, 2), "Positive": round(proba[1]*100, 2)})
            st.markdown("---")
            st.markdown("**📝 Review Summary:**")
            st.info(summary)
        else:
            st.warning("Please enter some text to analyze.")

elif choice == "Batch Upload":
    uploaded_file = st.file_uploader("Upload a CSV file (with a 'Text' column)", type=["csv"])

    # Decode predicted sentiment to string labels before displaying
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "Text" not in df.columns:
            st.error("CSV must contain a 'Text' column!")
        else:
            with st.spinner("Batch analyzing..."):
                df["Clean_Text"] = df["Text"].apply(clean_text)
                vecs = vectorizer.transform(df["Clean_Text"])
                preds = model.predict(vecs)

                # Decode predictions to string labels
                df["Predicted_Sentiment"] = ["positive" if pred == 1 else "negative" for pred in preds]

                # Optimized batch summarization logic
                summaries = []
                for text in df["Text"]:
                    summary = summarize_text(text)
                    summaries.append(summary)

                df["Summary"] = summaries

            st.success("✅ Batch prediction complete!")
            st.write(df[["Text", "Predicted_Sentiment", "Summary"]])

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", data=csv, file_name="batch_predictions.csv", mime="text/csv")

elif choice == "Model and Dataset":
    st.subheader("📊 Model Performance (on validation set)")
    st.write("""
    - Model: Voting Ensemble (Logistic Regression + eXtreme Gradient Boosting + Random Forest)
    - Accuracy: ~96%
    - Preprocessing: Cleaning + Stopword Removal (kept 'not') + TF-IDF (1-3 grams)
    - Summary Model: Hugging Face Inference API (facebook/bart-large-cnn)
    """)

    st.markdown("---")

    # Display additional performance metrics
    st.subheader("📈 Detailed Performance Metrics")

    # Display classification report as a table
    if os.path.exists(REPORT_PATH):
        report_df = pd.read_csv(REPORT_PATH)
        st.subheader("Classification Report")
        st.dataframe(report_df)  # Use st.dataframe() to display as a table
    else:
        st.warning("Classification report not found.")

    # Display confusion matrix image if available
    if os.path.exists(CONF_MATRIX_IMG):
        st.image(CONF_MATRIX_IMG, caption="Confusion Matrix")
    else:
        st.warning("Confusion matrix image not found.")

    st.markdown("---")

    st.subheader("📈 Visuals from Data Analysis of Dataset")
    # Display additional visuals if available
    visuals_dir = "visualizations"
    visual_files = [
        ("sentiment_distribution.png", "Sentiment Distribution"),
        ("sentiment_pie_chart.png", "Sentiment Pie Chart"),
        ("review_length_distribution.png", "Review Length Distribution"),
        ("positive_wordcloud.png", "Positive Word Cloud"),
        ("negative_wordcloud.png", "Negative Word Cloud")
    ]

    for file_name, caption in visual_files:
        path = os.path.join(visuals_dir, file_name)
        if os.path.exists(path):
            st.image(path, caption=caption)
        else:
            st.warning(f"⚠️ Visual not found: {file_name}")
