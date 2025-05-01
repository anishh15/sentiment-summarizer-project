import nltk
import re
import time
import requests
import os
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
stop_words.discard("not")  # Keep "not" to preserve sentiment

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGING_FACE_API_KEY = os.getenv("HF_TOKEN")  # Load API key from environment variable

def summarize_text(text, max_input_length=512, max_output_length=50):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": max_output_length, "min_length": 10, "do_sample": False},
    }

    for attempt in range(3):  # Retry up to 3 times
        try:
            response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    return result[0]["summary_text"]
                else:
                    return "Unexpected API response format."
            else:
                return f"Summarization API error: {response.status_code}, {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"
        time.sleep(2)  # Wait 2 seconds before retrying

    return "Error in summarization: API unavailable after multiple attempts"