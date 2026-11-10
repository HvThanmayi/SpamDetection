"""
Machine Learning Training Pipeline for YouTube Comment Spam Detection.

Trains, evaluates, and compares:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Support Vector Machine (LinearSVC with Probability Calibration)
4. Random Forest Classifier

Saves the best-performing pipeline using Joblib into `models/spam_detection_model.joblib`
and records actual evaluation metrics and diagnostic plots.
"""

import os
import sys
import glob
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import clean_text
from src.evaluation import (
    evaluate_predictions,
    build_comparison_dataframe,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_class_distribution,
    save_evaluation_results
)


# Configuration constants
DATASET_DIR = "dataset"
MODELS_DIR = "models"
COMBINED_CSV_PATH = os.path.join(DATASET_DIR, "youtube_comments.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "spam_detection_model.joblib")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_prepare_dataset(dataset_dir: str = DATASET_DIR) -> pd.DataFrame:
    """
    Loads YouTube spam comments from either the combined CSV or multiple artist CSV files.
    Cleans duplicates and null values, and saves the consolidated dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'CONTENT' and 'CLASS' columns.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    csv_files = sorted([
        f for f in glob.glob(os.path.join(dataset_dir, "Youtube*.csv"))
        if not os.path.basename(f).lower().startswith("youtube_comments")
    ])

    # Check if individual files exist
    if csv_files:
        print(f"[1/6] Discovered {len(csv_files)} artist dataset files:")
        dfs = []
        for f in csv_files:
            file_df = pd.read_csv(f)
            print(f"      - {os.path.basename(f)}: {len(file_df)} comments")
            dfs.append(file_df)
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.exists(COMBINED_CSV_PATH):
        print(f"[1/6] Loading consolidated dataset from {COMBINED_CSV_PATH}...")
        df = pd.read_csv(COMBINED_CSV_PATH)
    else:
        raise FileNotFoundError(
            f"No dataset files found in '{dataset_dir}'. Please ensure UCI YouTube spam CSV files "
            f"or '{COMBINED_CSV_PATH}' exist."
        )

    # Standardize column names to uppercase
    df.columns = [c.upper().strip() for c in df.columns]

    if "CONTENT" not in df.columns or "CLASS" not in df.columns:
        raise ValueError("Dataset must contain 'CONTENT' (comment text) and 'CLASS' (0 or 1) columns.")

    total_raw = len(df)
    # Drop records with null comment content or missing class label
    df = df.dropna(subset=["CONTENT", "CLASS"]).copy()

    # Cast CLASS to integer
    df["CLASS"] = df["CLASS"].astype(int)

    # Handle duplicates in comments
    duplicates_count = df.duplicated(subset=["CONTENT"]).sum()
    df = df.drop_duplicates(subset=["CONTENT"]).reset_index(drop=True)

    print(f"\n[Dataset Statistics]")
    print(f"  Total raw comments:      {total_raw}")
    print(f"  Duplicate comments dropped: {duplicates_count}")
    print(f"  Final clean comments:     {len(df)}")
    ham_count = (df['CLASS'] == 0).sum()
    spam_count = (df['CLASS'] == 1).sum()
    print(f"  - Legitimate (Ham, 0):    {ham_count} ({ham_count / len(df):.1%})")
    print(f"  - Spam (Spam, 1):         {spam_count} ({spam_count / len(df):.1%})")

    # Save consolidated dataset for simple subsequent access
    df.to_csv(COMBINED_CSV_PATH, index=False)
    print(f"  Consolidated dataset saved to: {COMBINED_CSV_PATH}")

    return df


def create_candidate_models():
    """
    Defines the four candidate machine learning classifiers.

    Notice: For LinearSVC, we wrap it in CalibratedClassifierCV. This preserves
    the linear margin optimization while adding calibrated predict_proba() support,
    allowing the Streamlit UI to display legitimate confidence percentages!
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine (LinearSVC)": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=RANDOM_STATE, dual="auto"),
            method="sigmoid",
            cv=3
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            random_state=RANDOM_STATE
        )
    }


def train_and_evaluate_all():
    """
    Executes the full training, evaluation, comparison, and export workflow.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("YOUTUBE COMMENT SPAM DETECTION - ML TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    df = load_and_prepare_dataset(DATASET_DIR)

    # 2. Save class distribution plot
    class_dist_plot_path = os.path.join(MODELS_DIR, "class_distribution.png")
    plot_class_distribution(df["CLASS"], save_path=class_dist_plot_path)
    print(f"[2/6] Class distribution chart saved: {class_dist_plot_path}")

    # 3. Text Preprocessing on entire corpus
    print("[3/6] Applying NLP text cleaning (URL tokenization, lowercase, normalization)...")
    # Clean the comments
    df["CLEAN_CONTENT"] = df["CONTENT"].apply(clean_text)

    # Drop any edge case where cleaned comment becomes empty
    valid_mask = df["CLEAN_CONTENT"].str.strip().str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    X = np.array(df["CLEAN_CONTENT"].tolist(), dtype=object)
    y = np.array(df["CLASS"].tolist(), dtype=int)

    # 4. Stratified Train-Test Split (avoids data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"\n[4/6] Data Split Completed (Stratified, random_state={RANDOM_STATE}):")
    print(f"      - Training Set: {len(X_train)} comments")
    print(f"      - Testing Set : {len(X_test)} comments (unseen evaluation)")

    # 5. Define candidate models and train pipelines
    models = create_candidate_models()
    evaluation_results = []
    trained_pipelines = {}

    print(f"\n[5/6] Training and Evaluating {len(models)} Algorithms...")
    print("-" * 70)

    for name, classifier in models.items():
        # Scikit-learn Pipeline ensures TF-IDF is fitted ONLY on training split
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),        # Unigrams & bigrams (e.g., 'subscribe', 'check out')
                max_features=4000,         # Top informative n-grams
                min_df=2,                  # Ignore terms that appear only once
                sublinear_tf=True          # Apply 1 + log(tf) scaling to dampen frequent words
            )),
            ("clf", classifier)
        ])

        # Train on training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Predict on unseen test set
        y_pred = pipeline.predict(X_test)

        # Evaluate metrics
        eval_metrics = evaluate_predictions(y_test, y_pred, model_name=name)
        evaluation_results.append(eval_metrics)

        # Generate and save confusion matrix heatmap
        cm_safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_path = os.path.join(MODELS_DIR, f"cm_{cm_safe_name}.png")
        plot_confusion_matrix(np.array(eval_metrics["confusion_matrix"]), model_name=name, save_path=cm_path)

        print(f"Model: {name:<35} | Acc: {eval_metrics['accuracy']:.4f} | "
              f"Prec: {eval_metrics['precision']:.4f} | Rec: {eval_metrics['recall']:.4f} | "
              f"F1: {eval_metrics['f1_score']:.4f} | FP: {eval_metrics['false_positives']}")

    # 6. Model Comparison Table
    comparison_df = build_comparison_dataframe(evaluation_results)
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON SUMMARY (Ranked by F1-Score)")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Save model comparison bar chart
    comparison_chart_path = os.path.join(MODELS_DIR, "model_comparison_chart.png")
    plot_model_comparison(comparison_df, save_path=comparison_chart_path)
    print(f"\nModel comparison chart saved: {comparison_chart_path}")

    # Identify Best Model:
    # We sort primarily by F1-Score, and break ties with lowest False Positives
    # (since erroneously blocking legitimate comments is bad user experience!)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = next(r for r in evaluation_results if r["model_name"] == best_model_name)

    print("\n" + "=" * 70)
    print(f"WINNING MODEL SELECTED: {best_model_name}")
    print(f"  - Accuracy:  {best_metrics['accuracy']:.2%}")
    print(f"  - Precision: {best_metrics['precision']:.2%}")
    print(f"  - Recall:    {best_metrics['recall']:.2%}")
    print(f"  - F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"  - False Positives: {best_metrics['false_positives']} (Legitimate comments marked as spam)")
    print("=" * 70)

    # Save best model to disk
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"\n[6/6] Winning model pipeline exported: {BEST_MODEL_PATH}")

    # Package metadata and save metrics JSON
    export_data = {
        "best_model_name": best_model_name,
        "dataset_stats": {
            "total_comments": len(df),
            "ham_count": int((df['CLASS'] == 0).sum()),
            "spam_count": int((df['CLASS'] == 1).sum()),
            "train_size": len(X_train),
            "test_size": len(X_test)
        },
        "comparison_table": comparison_df.to_dict(orient="records"),
        "models_evaluation": evaluation_results
    }
    save_evaluation_results(export_data, METRICS_JSON_PATH)
    print(f"Evaluation metrics JSON saved: {METRICS_JSON_PATH}")
    print("\nTraining workflow completed successfully!")

    return export_data


if __name__ == "__main__":
    train_and_evaluate_all()
