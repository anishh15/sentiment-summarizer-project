# data_analysis.py – Exploratory Analysis & Visualization (matplotlib + seaborn)

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_PATH, VISUALS_DIR, CLASSIFICATION_REPORT_PATH, CONFUSION_MATRIX_PATH

# Setup directories
os.makedirs(VISUALS_DIR, exist_ok=True)

# Updated paths for classification report and confusion matrix
classification_report_path = CLASSIFICATION_REPORT_PATH
confusion_matrix_path = CONFUSION_MATRIX_PATH

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully for analysis!")

# Clean data
df.dropna(inplace=True)
df = df[df['Score'].apply(lambda x: str(x).isdigit())]
df["Score"] = df["Score"].astype(int)
df = df[df["Score"] != 3].copy()
df["sentiment"] = df["Score"].apply(lambda s: "positive" if s > 3 else "negative")

# Sample data to improve speed
df = df.sample(5000, random_state=42)

# Step 5: Sentiment distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sentiment", hue="sentiment", palette="Set2", legend=False)
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "sentiment_distribution.png"))
plt.close()

# Step 5.1: Pie chart for sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
plt.title("Sentiment Distribution Pie Chart", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "sentiment_pie_chart.png"))
plt.close()

# Step 6: Review length distribution
df["review_length"] = df["Text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="review_length", hue="sentiment", bins=50, kde=True, palette="Set2")
plt.title("Review Length Distribution by Sentiment", fontsize=14)
plt.xlabel("Number of Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "review_length_distribution.png"))
plt.close()

# Step 7: Word clouds for positive and negative reviews
positive_texts = df[df.sentiment == "positive"]["Text"].dropna()
negative_texts = df[df.sentiment == "negative"]["Text"].dropna()

text_positive = " ".join(positive_texts.sample(min(2000, len(positive_texts)), random_state=42))
text_negative = " ".join(negative_texts.sample(min(2000, len(negative_texts)), random_state=42))

wc_positive = WordCloud(width=800, height=400, stopwords=stop_words, background_color='white', colormap='Greens').generate(text_positive)
plt.figure(figsize=(10, 5))
plt.imshow(wc_positive, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews Word Cloud", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "positive_wordcloud.png"))
plt.close()

wc_negative = WordCloud(width=800, height=400, stopwords=stop_words, background_color='white', colormap='Reds').generate(text_negative)
plt.figure(figsize=(10, 5))
plt.imshow(wc_negative, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews Word Cloud", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "negative_wordcloud.png"))
plt.close()
