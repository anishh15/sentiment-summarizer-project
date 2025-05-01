import os
import time
import pandas as pd
import re
import logging
import nltk
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.discard("not")  # Keep "not" for sentiment meaning

# Paths setup
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset loaded successfully!")
except FileNotFoundError:
    logger.error(f"File not found at {DATA_PATH}")
    exit(1)
except pd.errors.EmptyDataError:
    logger.error("File is empty or corrupted.")
    exit(1)

# Limit dataset size for testing (optional)
#df = df.sample(n=50000, random_state=42)  # Use only 50,000 samples for faster training

# Clean & preprocess
df.dropna(inplace=True)
df = df[df['Score'].apply(lambda x: str(x).isdigit())]
df["Score"] = df["Score"].astype(int)
df = df[df["Score"] != 3].copy()
df["sentiment"] = df["Score"].apply(lambda s: "positive" if s > 3 else "negative")

# Clean text and remove stopwords
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text if cleaned_text else "empty"

df["Text_clean"] = df["Text"].apply(clean_text)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["sentiment"])  # Converts 'negative' -> 0, 'positive' -> 1

# TF-IDF vectorization with reduced features
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 3))
X_vec = vectorizer.fit_transform(df["Text_clean"])

# Apply SMOTE to the entire dataset
logger.info("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vec, y_encoded)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
logger.info("Stratified train-test split done")

# Show class distribution after split
logger.info("Train class distribution:")
logger.info(pd.Series(y_train).value_counts())
logger.info("Test set class distribution (after SMOTE):")
logger.info(pd.Series(y_test).value_counts())

# Define classifiers
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Reduced number of trees to 50

# Train classifiers with progress tracking
classifiers = [('lr', lr), ('xgb', xgb), ('rf', rf)]
trained_classifiers = []

logger.info("Training individual classifiers with progress tracking...")
for name, clf in tqdm(classifiers, desc="Training Classifiers"):
    clf.fit(X_train, y_train)
    trained_classifiers.append((name, clf))

# Create soft voting ensemble
ensemble = VotingClassifier(estimators=trained_classifiers, voting='soft')
logger.info("Training ensemble model...")
ensemble.fit(X_train, y_train)
logger.info("Ensemble training complete")

# Evaluate
y_pred = ensemble.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)  # Decode predictions
logger.info("\nSentiment Classification Report (Ensemble):\n")
logger.info(classification_report(label_encoder.inverse_transform(y_test), y_pred_decoded))
logger.info(f"Balanced Accuracy: {balanced_accuracy_score(label_encoder.inverse_transform(y_test), y_pred_decoded)}")
logger.info(f"F1 Score (positive): {f1_score(label_encoder.inverse_transform(y_test), y_pred_decoded, pos_label='positive')}")
logger.info(f"F1 Score (negative): {f1_score(label_encoder.inverse_transform(y_test), y_pred_decoded, pos_label='negative')}")

# Save classification report and confusion matrix
report = classification_report(label_encoder.inverse_transform(y_test), y_pred_decoded, output_dict=True)
pd.DataFrame(report).to_csv(os.path.join(MODELS_DIR, "classification_report.csv"))
cm = ConfusionMatrixDisplay.from_predictions(label_encoder.inverse_transform(y_test), y_pred_decoded)
cm.plot()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))

# Test analyzer
def analyze_review(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    sentiment = label_encoder.inverse_transform(ensemble.predict(vec))[0]
    return sentiment  # Only return sentiment

logger.info("\nRunning a test example...\n")
sample = df["Text"].iloc[0]
sentiment = analyze_review(sample)
logger.info(f"Review:\n{sample}")
logger.info(f"\nSentiment: {sentiment}")

# Save model + vectorizer
joblib.dump(ensemble, os.path.join(MODELS_DIR, "sentiment_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
logger.info("\nModel and vectorizer saved to 'models/' folder.")