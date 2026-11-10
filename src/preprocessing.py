"""
Text Preprocessing Module for YouTube Comment Spam Detection.

This module provides functions to clean, normalize, and preprocess raw
YouTube comments before feeding them into the TF-IDF Vectorizer and ML models.
"""

import re
import html
import unicodedata
from typing import List, Optional


# Critical spam-indicator words that should NEVER be removed by stopword filtering
# In spam detection, standard stopwords like 'free', 'win', 'you', 'my' often carry strong intent signals.
PRESERVED_SPAM_KEYWORDS = {
    "free", "win", "gift", "giveaway", "subscribe", "sub", "channel",
    "check", "click", "link", "money", "dollar", "cash", "offer", "discount",
    "promo", "code", "visit", "watch", "please", "share", "like", "follow",
    "httpurl", "emailaddr"
}

# Standard English stopwords (compiled without external network downloads for maximum reliability)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}

# Effective stopwords = Default stopwords MINUS preserved spam triggers
EFFECTIVE_STOPWORDS = DEFAULT_STOPWORDS - PRESERVED_SPAM_KEYWORDS


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Cleans and standardizes raw text from YouTube comments.

    Pipeline Steps:
    1. Null & Type Handling: Ensure input is a valid string.
    2. HTML Unescaping: Decodes HTML entities commonly found in web comments (e.g. &amp; -> &, &quot; -> ").
    3. URL Normalization: Replaces web links (http, https, www, t.co, bit.ly, etc.) with a token 'httpurl'.
       This preserves the crucial spam indicator of containing a URL without overfitting to domain names.
    4. Email Normalization: Replaces email patterns with 'emailaddr'.
    5. Case Normalization: Converts text to lowercase.
    6. Special Characters & Punctuation: Removes non-alphanumeric noise while keeping spaces and token boundaries.
    7. Whitespace Collapsing: Removes redundant spaces, tabs, and newlines.
    8. Token Filtering (Optional): Drops general stopwords while strictly preserving spam indicators.

    Args:
        text (str): Raw YouTube comment text.
        remove_stopwords (bool): Whether to remove generic stopwords. Defaults to False,
                                 allowing TF-IDF n-grams to capture phrases naturally.

    Returns:
        str: Cleaned, normalized comment string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Step 1: Decode HTML entities
    cleaned = html.unescape(text)

    # Step 2: Normalize unicode (handles accent variations and special unicode lookalikes)
    cleaned = unicodedata.normalize("NFKD", cleaned)

    # Step 3: Replace URLs with 'httpurl' token
    # Matches http://, https://, www., and common shorteners
    url_pattern = r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|org|net|io|co|ly|gl|be|tv|xyz)\S*)"
    cleaned = re.sub(url_pattern, " httpurl ", cleaned, flags=re.IGNORECASE)

    # Step 4: Replace email addresses with 'emailaddr' token
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    cleaned = re.sub(email_pattern, " emailaddr ", cleaned, flags=re.IGNORECASE)

    # Step 5: Convert to lowercase
    cleaned = cleaned.lower()

    # Step 6: Remove special characters, preserving alphanumeric characters and underscores
    # We allow letters, numbers, and whitespaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # Step 7: Collapse redundant whitespaces
    tokens = cleaned.split()

    # Step 8: Optional stopword filtering that protects critical promotional terms
    if remove_stopwords:
        tokens = [word for word in tokens if word not in EFFECTIVE_STOPWORDS or word in PRESERVED_SPAM_KEYWORDS]

    return " ".join(tokens)


def preprocess_corpus(texts: List[str], remove_stopwords: bool = False) -> List[str]:
    """
    Preprocesses an entire list or Series of comments.

    Args:
        texts (List[str]): List of raw comment strings.
        remove_stopwords (bool): Whether to apply stopword removal.

    Returns:
        List[str]: List of cleaned comment strings.
    """
    return [clean_text(t, remove_stopwords=remove_stopwords) for t in texts]


def extract_spam_cues(raw_text: str) -> dict:
    """
    Extracts heuristic rule cues from a comment for educational & diagnostic display.
    This helps students and reviewers understand WHY a comment looks suspicious.

    Args:
        raw_text (str): Raw comment text.

    Returns:
        dict: Diagnostic indicators found in the text.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "has_url": False,
            "has_promotional_keyword": False,
            "excessive_caps": False,
            "has_currency_or_reward": False,
            "detected_keywords": []
        }

    has_url = bool(re.search(r"(https?://|www\.|\b\w+\.(?:com|org|io|ly|be))", raw_text, re.IGNORECASE))

    # Check for excessive uppercase letters (e.g. "FREE IPHONE CLICK HERE")
    letters = [c for c in raw_text if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    excessive_caps = len(letters) > 10 and (len(caps) / len(letters)) > 0.45

    # Check currency symbols or reward promises
    reward_pattern = r"(\$|\b(usd|euro|inr|cash|gift|card|giveaway|prize|won|win)\b)"
    has_reward = bool(re.search(reward_pattern, raw_text, re.IGNORECASE))

    # Detect spam words
    lower = raw_text.lower()
    cues_detected = []
    sample_cues = ["subscribe", "sub to", "check my", "check out", "free", "win", "giveaway", "money", "click", "channel"]
    for cue in sample_cues:
        if cue in lower:
            cues_detected.append(cue)

    return {
        "has_url": has_url,
        "has_promotional_keyword": len(cues_detected) > 0,
        "excessive_caps": excessive_caps,
        "has_currency_or_reward": has_reward,
        "detected_keywords": cues_detected
    }


if __name__ == "__main__":
    # Demonstration and verification of preprocessing function
    test_samples = [
        "Check out my new song here: https://youtube.com/watch?v=12345 &amp; subscribe!!",
        "Congratulations! You have WON a free iPhone 15! Click http://bit.ly/claim-now",
        "This song is amazing, listening to it in 2024 brings back so many memories &lt;3",
        "Please sub to my channel guys! I make daily gaming videos and give free robux!",
        "Great video as always! Very insightful and well explained."
    ]

    print("=" * 70)
    print("TEXT PREPROCESSING DEMONSTRATION")
    print("=" * 70)
    for idx, sample in enumerate(test_samples, 1):
        cleaned = clean_text(sample)
        cues = extract_spam_cues(sample)
        print(f"\n[Sample {idx}]")
        print(f"  Raw Text     : {sample}")
        print(f"  Cleaned Text : {cleaned}")
        print(f"  Detected Cues: {cues}")
    print("=" * 70)
