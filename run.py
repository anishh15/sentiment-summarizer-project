from src.utils import clean_text, summarize_text
from src.config import MODELS_DIR
from src.predict_analyze import analyze_review

print("\n🔍 Enter a review below to analyze (or press Enter to use a sample):\n")
user_input = input("Your Review: ").strip()

if not user_input:
    user_input = """
    I’ve tried many protein bars and most are either too sweet or have a weird texture. This one was perfect.
    Not only does it taste like chocolate, but it also keeps me full for hours. Absolutely loved it.
    """

sentiment, summary = analyze_review(user_input)  # No changes needed here as analyze_review is already updated

print("\n📌 Sentiment:", sentiment.capitalize())
print("\n📋 Summary:", summary)
