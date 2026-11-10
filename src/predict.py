"""
Prediction Module for YouTube Comment Spam Detection.

Loads the saved trained model pipeline and exposes simple, production-ready
functions for single comment prediction and batch CSV comment inference.
"""

import os
import sys
from typing import Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text, extract_spam_cues

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "spam_detection_model.joblib")

# Global in-memory cache for loaded model
_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the trained scikit-learn pipeline from disk.
    Ensures that the model is read into memory only once rather than re-loaded
    or re-trained on each user interaction.

    Args:
        model_path (str): File path to .joblib model file.

    Returns:
        Pipeline: Trained scikit-learn pipeline object.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run 'python src/train_model.py' first to train and export the model."
        )

    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def predict_comment(comment: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Performs spam classification on a single YouTube comment.

    Args:
        comment (str): Raw YouTube comment text entered by the user.
        model_path (str): Optional override path to model file.

    Returns:
        dict: Structured prediction output containing:
            - 'original_comment': raw input text
            - 'cleaned_comment': normalized text fed to vectorizer
            - 'prediction': 1 for SPAM, 0 for NOT SPAM
            - 'label': human-readable string ("SPAM" or "NOT SPAM")
            - 'confidence': percentage confidence (0.0 to 100.0)
            - 'spam_probability': float probability of being spam (0.0 to 1.0)
            - 'ham_probability': float probability of being ham (0.0 to 1.0)
            - 'indicators': diagnostic heuristic cues (URLs, caps, spam words)
            - 'explanation': simple English explanation for students/users
    """
    if not isinstance(comment, str) or not comment.strip():
        return {
            "original_comment": "" if comment is None else str(comment),
            "cleaned_comment": "",
            "prediction": 0,
            "label": "NOT SPAM",
            "confidence": 0.0,
            "spam_probability": 0.0,
            "ham_probability": 1.0,
            "indicators": {},
            "explanation": "Empty or whitespace comment provided. Defaulted to NOT SPAM."
        }

    # Step 1: Text preprocessing
    cleaned = clean_text(comment)
    indicators = extract_spam_cues(comment)

    # Edge case: if cleaning removed everything (e.g. only symbols)
    if not cleaned.strip():
        cleaned = "empty"

    # Step 2: Load model pipeline
    pipeline = load_model(model_path)

    # Step 3: Run prediction
    pred_class = int(pipeline.predict([cleaned])[0])

    # Step 4: Confidence calculation
    # Support calibrated probabilities if classifier provides predict_proba
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        ham_prob = float(probs[0])
        spam_prob = float(probs[1])
        confidence = float(probs[pred_class] * 100.0)
    elif hasattr(pipeline, "decision_function"):
        # For raw uncalibrated SVMs, calculate confidence via sigmoid on distance
        df_val = float(pipeline.decision_function([cleaned])[0])
        spam_prob = 1.0 / (1.0 + np.exp(-df_val))
        ham_prob = 1.0 - spam_prob
        confidence = float((spam_prob if pred_class == 1 else ham_prob) * 100.0)
    else:
        spam_prob = float(pred_class)
        ham_prob = 1.0 - spam_prob
        confidence = 90.0

    label_str = "SPAM" if pred_class == 1 else "NOT SPAM"

    # Step 5: Generate clear explanation
    if pred_class == 1:
        reasons = []
        if indicators.get("has_url"):
            reasons.append("contains promotional links or external URLs")
        if indicators.get("detected_keywords"):
            reasons.append(f"contains promotional spam keywords: {', '.join(indicators['detected_keywords'])}")
        if indicators.get("has_currency_or_reward"):
            reasons.append("offers giveaways, money, or prizes")
        if indicators.get("excessive_caps"):
            reasons.append("uses excessive uppercase shouting")

        if reasons:
            explanation = f"Flagged as SPAM because the comment {' and '.join(reasons)}."
        else:
            explanation = "Flagged as SPAM due to stylistic and vocabulary resemblance to historical YouTube promotional spam."
    else:
        explanation = "Classified as NOT SPAM (Legitimate). The comment exhibits natural language patterns typical of genuine viewers and discussions."

    return {
        "original_comment": comment,
        "cleaned_comment": cleaned,
        "prediction": pred_class,
        "label": label_str,
        "confidence": round(confidence, 2),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
        "indicators": indicators,
        "explanation": explanation
    }


def predict_batch(
    comments_input: Union[List[str], pd.DataFrame, pd.Series],
    comment_col: str = None,
    model_path: str = DEFAULT_MODEL_PATH
) -> pd.DataFrame:
    """
    Performs spam classification on a collection of comments (List or DataFrame).

    Args:
        comments_input: List of comments or a Pandas DataFrame.
        comment_col: Name of column containing comments if DataFrame provided.
        model_path: Optional override path to model file.

    Returns:
        pd.DataFrame: DataFrame with original comment, cleaned text, prediction (0/1),
                      label (SPAM / NOT SPAM), confidence (%), and spam probability.
    """
    if isinstance(comments_input, pd.DataFrame):
        df = comments_input.copy()
        if comment_col is None:
            # Auto-detect comment column
            candidates = ["content", "comment", "comments", "text", "body", "message"]
            for col in df.columns:
                if col.lower() in candidates:
                    comment_col = col
                    break
            if comment_col is None:
                # Default to first object/string column
                string_cols = df.select_dtypes(include=["object"]).columns
                if len(string_cols) > 0:
                    comment_col = string_cols[0]
                else:
                    comment_col = df.columns[0]

        raw_comments = df[comment_col].astype(str).tolist()
    elif isinstance(comments_input, (list, tuple, pd.Series)):
        raw_comments = [str(c) for c in comments_input]
        df = pd.DataFrame({"Comment": raw_comments})
        comment_col = "Comment"
    else:
        raise ValueError("comments_input must be a List, Series, or DataFrame.")

    # Clean comments
    cleaned_comments = [clean_text(c) for c in raw_comments]
    # Handle empties
    safe_cleaned = [c if c.strip() else "empty" for c in cleaned_comments]

    # Predict with model pipeline
    pipeline = load_model(model_path)
    predictions = pipeline.predict(safe_cleaned).astype(int)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(safe_cleaned)
        spam_probs = probabilities[:, 1]
        confidences = np.max(probabilities, axis=1) * 100.0
    else:
        spam_probs = predictions.astype(float)
        confidences = np.full(len(predictions), 90.0)

    # Attach results to dataframe
    df["Cleaned_Comment"] = cleaned_comments
    df["Prediction"] = predictions
    df["Label"] = ["SPAM" if p == 1 else "NOT SPAM" for p in predictions]
    df["Confidence (%)"] = np.round(confidences, 2)
    df["Spam_Probability"] = np.round(spam_probs, 4)

    return df


if __name__ == "__main__":
    # Test sample comments
    sample_comments = [
        "Congratulations! You have won a free iPhone. Click this link now! http://bit.ly/claim",
        "This video is amazing. Thank you for sharing!",
        "Please check out my channel and subscribe! I post daily music beats.",
        "I really loved the explanation around minute 3:15, great teaching!",
        "Earn $500 daily from home! Message me on WhatsApp +123456789"
    ]

    print("=" * 70)
    print("PREDICTION MODULE INFERENCE TEST")
    print("=" * 70)
    for c in sample_comments:
        res = predict_comment(c)
        print(f"\nComment:    {res['original_comment']}")
        print(f"Prediction: {res['label']} ({res['confidence']}% confidence)")
        print(f"Explanation: {res['explanation']}")
    print("=" * 70)
