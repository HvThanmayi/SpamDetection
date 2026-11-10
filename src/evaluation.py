"""
Model Evaluation and Visualization Module.

Provides functions to compute classification metrics (Accuracy, Precision,
Recall, F1-score), generate confusion matrices, produce comparative charts,
and explain evaluation concepts simply for students.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating figures safely
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics for Spam Detection.

    Definitions:
    - Class 1 = Spam (Positive)
    - Class 0 = Ham / Legitimate (Negative)

    Metrics:
    - Accuracy: Overall correct predictions / total predictions.
    - Precision: Of all comments flagged as spam, how many were actually spam?
                 High precision = very few legitimate comments mistakenly blocked (Low False Positives).
    - Recall: Of all actual spam comments, how many did the model catch?
              High recall = few spam comments slipped through (Low False Negatives).
    - F1-Score: Harmonic mean of precision and recall. Best single metric for balanced assessment.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        model_name: Label identifier for the model.

    Returns:
        dict: Evaluated metrics and confusion matrix breakdown.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # cm layout in sklearn:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }


def build_comparison_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a clean comparison DataFrame from multiple evaluation dictionaries.

    Args:
        results: List of evaluation metric dictionaries.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1_score"], 4),
            "False Positives (Ham -> Spam)": r["false_positives"],
            "False Negatives (Spam -> Ham)": r["false_negatives"]
        })
    df = pd.DataFrame(rows)
    return df.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str = None) -> plt.Figure:
    """
    Plots a clear, professional heatmap for the confusion matrix.

    Args:
        cm: 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        model_name: Name of the evaluated algorithm.
        save_path: Optional path to save image file.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set_theme(style="white")

    group_names = ["True Negative\n(Ham correctly kept)",
                   "False Positive\n(Ham wrongly blocked)",
                   "False Negative\n(Spam missed)",
                   "True Positive\n(Spam caught)"]
    group_counts = [f"{v:0.0f}" for v in np.array(cm).flatten()]
    group_percentages = [f"{v:.1%}" for v in np.array(cm).flatten() / np.sum(cm)]

    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        linecolor="#f0f0f0",
        xticklabels=["Predicted Ham (0)", "Predicted Spam (1)"],
        yticklabels=["Actual Ham (0)", "Actual Spam (1)"],
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"}
    )

    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Model Prediction", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Label", fontsize=11, labelpad=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Plots a multi-metric bar chart comparing evaluated algorithms.

    Args:
        comparison_df: DataFrame with Model, Accuracy, Precision, Recall, F1-Score.
        save_path: Optional path to save image file.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Reshape for seaborn grouped barplot
    melted = pd.melt(comparison_df, id_vars=["Model"], value_vars=metrics,
                     var_name="Metric", value_name="Score")

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette, ax=ax)

    ax.set_ylim(0.75, 1.02)
    ax.set_title("Machine Learning Algorithm Comparison", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Classification Algorithm", fontsize=11, labelpad=8)
    ax.set_ylabel("Evaluation Score (0.0 to 1.0)", fontsize=11, labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Metric", loc="lower right", framealpha=0.9)

    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        fontsize=8, color='black',
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_class_distribution(y_series: pd.Series, save_path: str = None) -> plt.Figure:
    """
    Plots the distribution of Spam vs Ham in the dataset.

    Args:
        y_series: Series of class labels (0 and 1).
        save_path: Optional path to save image file.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = y_series.value_counts().rename(index={0: "Ham / Legitimate (0)", 1: "Spam (1)"})

    colors = ["#2ca02c", "#d62728"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="#333", linewidth=1)

    ax.set_title("YouTube Comments Class Distribution", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Number of Comments", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h} ({h / len(y_series):.1%})",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=10, weight="bold",
                    xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def save_evaluation_results(results: Dict[str, Any], filepath: str):
    """Saves evaluation results dict to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Loads evaluation results dict from JSON file."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
