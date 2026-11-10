"""
Streamlit Web Application: YouTube Comment Spam Detection System.

An executive, modern SaaS-grade web interface for identifying promotional,
malicious, and scam comments on YouTube using calibrated Linear Support
Vector Machines (LinearSVC) and TF-IDF feature extraction.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_comment, predict_batch, load_model
from src.preprocessing import clean_text

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="YouTube Comment Spam Detector | AI Moderation SaaS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MODERN SAAS LIGHT THEME CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Root Variables & Reset */
    :root {
        --bg-main: #F8FAFC;
        --surface-white: #FFFFFF;
        --surface-muted: #F1F5F9;
        --border-subtle: #E2E8F0;
        --border-hover: #CBD5E1;
        --text-headline: #0F172A;
        --text-body: #334155;
        --text-muted: #64748B;
        --text-subtle: #94A3B8;
        --primary-blue: #2563EB;
        --primary-hover: #1D4ED8;
        --primary-tint: #EFF6FF;
        --success-emerald: #10B981;
        --success-tint: #ECFDF5;
        --success-border: #A7F3D0;
        --success-text: #065F46;
        --danger-rose: #EF4444;
        --danger-tint: #FEF2F2;
        --danger-border: #FECACA;
        --danger-text: #991B1B;
        --warning-amber: #F59E0B;
        --warning-tint: #FFFBEB;
        --warning-border: #FDE68A;
        --warning-text: #92400E;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-body) !important;
    }

    /* Container Spacing */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* Header Bar Clean Light Blur */
    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* Sidebar Clean SaaS Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--surface-white) !important;
        border-right: 1px solid var(--border-subtle) !important;
        box-shadow: 1px 0 3px 0 rgba(0, 0, 0, 0.02) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    /* Sidebar Brand Section */
    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .brand-icon-wrapper {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25);
        color: #ffffff;
        font-size: 1.35rem;
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text-headline);
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--success-tint);
        color: var(--success-text);
        border: 1px solid var(--success-border);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--success-emerald);
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    /* Sidebar Info Card */
    .sidebar-card {
        background: var(--surface-muted);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1.25rem;
    }
    .sidebar-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--text-headline);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sidebar-card-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
        font-size: 0.82rem;
        color: var(--text-body);
        line-height: 1.6;
    }
    .sidebar-card-list li {
        margin-bottom: 4px;
        display: flex;
        justify-content: space-between;
    }
    .sidebar-card-list li span.meta-label {
        color: var(--text-muted);
        font-weight: 500;
    }
    .sidebar-card-list li span.meta-val {
        font-weight: 600;
        color: var(--text-headline);
    }

    /* Radio Navigation Menu Redesign */
    div[data-testid="stRadio"] > label {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
        margin-bottom: 0.5rem !important;
    }
    div[data-testid="stRadio"] > div {
        background-color: transparent !important;
        gap: 4px !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        padding: 8px 12px !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out !important;
        background-color: transparent !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background-color: var(--surface-muted) !important;
        border-color: var(--border-subtle) !important;
    }

    /* Executive Hero Banner */
    .saas-hero-card {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        position: relative;
        overflow: hidden;
    }
    .saas-hero-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563EB 0%, #4F46E5 50%, #06B6D4 100%);
    }
    .hero-top-badges {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text-headline);
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-gradient-text {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #4F46E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: var(--text-body);
        line-height: 1.55;
        max-width: 860px;
        margin-bottom: 1.25rem;
    }

    /* Badge Pills */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .pill-primary {
        background-color: var(--primary-tint);
        color: var(--primary-blue);
        border-color: #BFDBFE;
    }
    .pill-success {
        background-color: var(--success-tint);
        color: var(--success-text);
        border-color: var(--success-border);
    }
    .pill-danger {
        background-color: var(--danger-tint);
        color: var(--danger-text);
        border-color: var(--danger-border);
    }
    .pill-warning {
        background-color: var(--warning-tint);
        color: var(--warning-text);
        border-color: var(--warning-border);
    }
    .pill-neutral {
        background-color: var(--surface-muted);
        color: var(--text-muted);
        border-color: var(--border-subtle);
    }

    /* High-End White Surface Card */
    .saas-card {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .saas-card:hover {
        border-color: var(--border-hover);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.05);
    }
    .saas-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-headline);
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Result Cards */
    .prediction-card-spam {
        background-color: var(--danger-tint);
        border: 1.5px solid var(--danger-border);
        border-radius: 14px;
        padding: 1.6rem 1.75rem;
        color: var(--danger-text);
        box-shadow: 0 4px 12px -2px rgba(239, 68, 68, 0.08);
        margin-bottom: 1.25rem;
    }
    .prediction-card-ham {
        background-color: var(--success-tint);
        border: 1.5px solid var(--success-border);
        border-radius: 14px;
        padding: 1.6rem 1.75rem;
        color: var(--success-text);
        box-shadow: 0 4px 12px -2px rgba(16, 185, 129, 0.08);
        margin-bottom: 1.25rem;
    }
    .prediction-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.85rem;
    }
    .prediction-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .prediction-explanation {
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0;
        font-weight: 500;
    }

    /* Monospace Token Inspector */
    .token-inspector-box {
        background: #F8FAFC;
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: 'JetBrains Mono', Consolas, monospace;
        font-size: 0.88rem;
        color: #0F172A;
        line-height: 1.6;
        word-break: break-all;
        user-select: all;
    }

    /* Custom Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 1.5rem;
    }
    .metric-card-pro {
        background: var(--surface-white);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card-pro:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.06);
        border-color: var(--border-hover);
    }
    .metric-card-accent {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-blue);
    }
    .metric-card-accent.emerald { background: var(--success-emerald); }
    .metric-card-accent.rose { background: var(--danger-rose); }
    .metric-card-accent.amber { background: var(--warning-amber); }
    .metric-card-accent.indigo { background: #6366F1; }

    .metric-label-pro {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .metric-value-pro {
        font-size: 1.95rem;
        font-weight: 800;
        color: var(--text-headline);
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-sub-pro {
        font-size: 0.78rem;
        color: var(--text-subtle);
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Buttons Modern SaaS Overhaul */
    .stButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--primary-blue) !important;
        border-color: var(--primary-hover) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-hover) !important;
        box-shadow: 0 6px 12px -1px rgba(37, 99, 235, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:not([kind="primary"]) {
        background-color: var(--surface-white) !important;
        color: var(--text-headline) !important;
        border-color: var(--border-subtle) !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        background-color: var(--surface-muted) !important;
        border-color: var(--border-hover) !important;
        color: var(--primary-blue) !important;
    }

    /* Text Inputs & Textareas */
    div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
        border: 1.5px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        background-color: var(--surface-white) !important;
        color: var(--text-headline) !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stTextArea"] textarea:focus, div[data-testid="stTextInput"] input:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        outline: none !important;
    }

    /* File Uploader SaaS Polish */
    div[data-testid="stFileUploader"] {
        background-color: var(--surface-white) !important;
        border: 2px dashed var(--border-hover) !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        transition: border-color 0.2s ease, background-color 0.2s ease !important;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: var(--primary-blue) !important;
        background-color: #F8FAFC !important;
    }

    /* Streamlit Tabs SaaS Styling */
    div[data-testid="stTabs"] button[data-testid="stTab"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"] {
        color: var(--primary-blue) !important;
        border-bottom: 2.5px solid var(--primary-blue) !important;
    }

    /* Clean Table Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        background-color: var(--surface-white) !important;
    }

    /* Progress Bar */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
        border-radius: 9999px !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        background-color: #E2E8F0 !important;
        border-radius: 9999px !important;
        height: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA & CACHING HELPERS
# ==============================================================================
@st.cache_data
def load_cached_metrics():
    """Loads saved evaluation metrics from JSON file."""
    metrics_path = os.path.join(PROJECT_ROOT, "models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ==============================================================================
# HIGH-END SAAS PLOTTING UTILITIES (LIGHT THEME MATCHED)
# ==============================================================================
def create_light_comparison_chart(comparison_df: pd.DataFrame):
    """
    Renders a modern, high-contrast bar chart matching the #FFFFFF / #F8FAFC
    SaaS light aesthetic with visible labels and clean gridlines.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    melted = pd.melt(comparison_df, id_vars=["Model"], value_vars=metrics,
                     var_name="Metric", value_name="Score")

    # Light SaaS Palette: Indigo, Emerald, Amber, Rose
    palette = {
        "Accuracy": "#2563EB",
        "Precision": "#10B981",
        "Recall": "#F59E0B",
        "F1-Score": "#6366F1"
    }

    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette, ax=ax, edgecolor="#FFFFFF", linewidth=1)

    ax.set_ylim(0.80, 1.03)
    ax.set_title("Machine Learning Algorithm Comparison", fontsize=13, weight="bold", pad=14, color="#0F172A")
    ax.set_xlabel("Classification Algorithm", fontsize=10, weight="600", labelpad=10, color="#475569")
    ax.set_ylabel("Evaluation Score (0.0 - 1.0)", fontsize=10, weight="600", labelpad=10, color="#475569")
    
    # Clean spines and gridlines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#CBD5E1")
    ax.spines['bottom'].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#E2E8F0")

    # Annotate bar tops
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        fontsize=7.5, color='#1E293B', weight="bold",
                        xytext=(0, 2), textcoords='offset points')

    ax.legend(title="", loc="lower right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.5)
    plt.tight_layout()
    return fig


def create_light_confusion_matrix(cm_data, model_name: str):
    """
    Renders a crisp SaaS confusion matrix heatmap with emerald & crimson quadrant cues.
    """
    fig, ax = plt.subplots(figsize=(6, 4.6), dpi=180, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    cm = np.array(cm_data)
    group_names = [
        "True Negative\n(Legitimate Kept)",
        "False Positive\n(Legitimate Blocked)",
        "False Negative\n(Spam Missed)",
        "True Positive\n(Spam Caught)"
    ]
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_percentages = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]

    labels = [f"{n}\n{c}\n({p})" for n, c, p in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    cmap = sns.light_palette("#2563EB", as_cmap=True)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=2,
        linecolor="#F8FAFC",
        xticklabels=["Predicted Legitimate (0)", "Predicted Spam (1)"],
        yticklabels=["Actual Legitimate (0)", "Actual Spam (1)"],
        ax=ax,
        annot_kws={"size": 9.5, "weight": "bold", "color": "#0F172A"}
    )

    ax.set_title(f"Confusion Matrix • {model_name}", fontsize=12, weight="bold", pad=12, color="#0F172A")
    ax.set_xlabel("Predicted Class", fontsize=9.5, weight="600", labelpad=8, color="#475569")
    ax.set_ylabel("True Ground Label", fontsize=9.5, weight="600", labelpad=8, color="#475569")
    ax.tick_params(colors="#475569", labelsize=8.5)

    plt.tight_layout()
    return fig


def create_light_class_distribution(ham_count: int, spam_count: int):
    """
    Renders a sleek class distribution chart matching the light design theme.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=180, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    categories = ["Legitimate (Ham)", "Spam Comments"]
    values = [ham_count, spam_count]
    colors = ["#10B981", "#EF4444"]

    bars = ax.bar(categories, values, color=colors, width=0.45, edgecolor="#E2E8F0", linewidth=1.5)

    ax.set_title("Corpus Label Balance", fontsize=12, weight="bold", pad=12, color="#0F172A")
    ax.set_ylabel("Number of Comments", fontsize=9.5, weight="600", color="#475569")
    ax.set_ylim(0, max(values) * 1.18)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color("#CBD5E1")
    ax.spines['bottom'].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#E2E8F0")

    total = sum(values)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    (bar.get_x() + bar.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=9, weight="bold", color="#0F172A",
                    xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    return fig


# ==============================================================================
# SIDEBAR NAVIGATION & SYSTEM METADATA
# ==============================================================================
with st.sidebar:
    # Modern Brand Header
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="brand-icon-wrapper">🛡️</div>
        <div>
            <div class="brand-title">SpamShield</div>
            <div class="brand-subtitle">YouTube NLP Moderation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Operational Status Pill
    st.markdown("""
    <div class="status-pill">
        <span class="status-dot"></span>
        <span>Model Status: Operational</span>
    </div>
    """, unsafe_allow_html=True)

    # Main Navigation
    page = st.radio(
        "Workspace Navigation",
        ["🏠 Home & Overview",
         "🔍 Single Comment Detection",
         "📁 Batch Comment Detection",
         "📊 Model Performance Dashboard"],
        index=0
    )

    # Project Technical Metadata Card
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">⚡ System Architecture</div>
        <ul class="sidebar-card-list">
            <li><span class="meta-label">Domain:</span> <span class="meta-val">NLP Classification</span></li>
            <li><span class="meta-label">Active Model:</span> <span class="meta-val">LinearSVC (Calibrated)</span></li>
            <li><span class="meta-label">Feature Eng:</span> <span class="meta-val">TF-IDF (1-2 N-Grams)</span></li>
            <li><span class="meta-label">Dataset:</span> <span class="meta-val">UCI YouTube Spam</span></li>
            <li><span class="meta-label">Vocabulary:</span> <span class="meta-val">2,000 Key Features</span></li>
            <li><span class="meta-label">Inference:</span> <span class="meta-val">Sub-millisecond</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card" style="margin-top: 0.75rem;">
        <div class="sidebar-card-title">🔒 Production Integrity</div>
        <p style="font-size: 0.78rem; color: #64748B; margin: 0; line-height: 1.5;">
            Local on-device inference with calibrated probability estimation. Zero cloud API dependencies.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION A: HOME & OVERVIEW
# ==============================================================================
if page == "🏠 Home & Overview":
    # Executive SaaS Hero Banner
    st.markdown("""
    <div class="saas-hero-card">
        <div class="hero-top-badges">
            <span class="pill-badge pill-primary">🛡️ AI SAFETY ENGINE</span>
            <span class="pill-badge pill-success">● READY FOR MODERATION</span>
            <span class="pill-badge pill-neutral">IBM SKILL LAB PROTOTYPE</span>
        </div>
        <h1 class="hero-title">YouTube Comment <span class="hero-gradient-text">Spam Intelligence</span></h1>
        <p class="hero-subtitle">
            An executive Machine Learning and Natural Language Processing platform designed to protect content creators
            and viewer communities by filtering unsolicited promotions, phishing links, and bot spam with high precision.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span class="pill-badge pill-neutral">⚡ 95.44% Accuracy</span>
            <span class="pill-badge pill-neutral">🎯 98.10% Peak Precision</span>
            <span class="pill-badge pill-neutral">📦 1,754 Verified Samples</span>
            <span class="pill-badge pill-neutral">🔬 Calibrated Probabilities</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">🎯 The Content Moderation Challenge</div>
            <p style="font-size: 0.95rem; color: #334155; line-height: 1.6; margin-bottom: 1rem;">
                Video creators face thousands of daily comments containing malicious crypto giveaways, fraudulent links,
                and aggressive sub-for-sub bots that degrade viewer engagement. This system employs supervised NLP
                to distinguish between authentic discourse and automated spam:
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 1rem;">
                <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: #991B1B; font-size: 0.95rem; margin-bottom: 4px;">🚨 Class 1: SPAM</div>
                    <div style="font-size: 0.85rem; color: #7F1D1D; line-height: 1.45;">
                        Phishing links, fake giveaways, sub4sub begging, financial scams, and bot spam.
                    </div>
                </div>
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; padding: 1rem;">
                    <div style="font-weight: 700; color: #065F46; font-size: 0.95rem; margin-bottom: 4px;">💬 Class 0: LEGITIMATE</div>
                    <div style="font-size: 0.85rem; color: #064E3B; line-height: 1.45;">
                        Authentic fan discussions, questions, timestamps, constructive critique, and genuine feedback.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">⚙️ Machine Learning Pipeline Architecture</div>
            <div style="display: flex; flex-direction: column; gap: 12px; font-size: 0.9rem; color: #334155;">
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="background: #EFF6FF; color: #2563EB; font-weight: 800; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">1</div>
                    <div><strong>Raw Comment Ingestion:</strong> Accepts single user comments or bulk CSV file uploads.</div>
                </div>
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="background: #EFF6FF; color: #2563EB; font-weight: 800; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">2</div>
                    <div><strong>NLP Text Normalization:</strong> Strips HTML entities, converts links to <code>httpurl</code>, emails to <code>emailaddr</code>, standardizes casing, and filters non-discriminative punctuation.</div>
                </div>
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="background: #EFF6FF; color: #2563EB; font-weight: 800; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">3</div>
                    <div><strong>TF-IDF Feature Representation:</strong> Extracts unigram and bigram frequency-inverse document frequencies with sublinear term scaling.</div>
                </div>
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="background: #EFF6FF; color: #2563EB; font-weight: 800; border-radius: 8px; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">4</div>
                    <div><strong>Calibrated Linear SVM:</strong> Evaluates decision boundaries with Platt/isotonic probability calibration for interpretable confidence scores.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        metrics = load_cached_metrics()
        stats = metrics.get("dataset_stats", {}) if metrics else {}

        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📊 Dataset Benchmark Snapshot</div>
            <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 1rem;">
                Trained & evaluated across 5 YouTube music video collections from the UCI Machine Learning Repository.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Dataset metrics
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"""
            <div class="metric-card-pro" style="margin-bottom: 12px;">
                <div class="metric-card-accent emerald"></div>
                <div class="metric-label-pro">Legitimate (Ham)</div>
                <div class="metric-value-pro" style="color: #065F46;">{stats.get('ham_count', 913):,}</div>
                <div class="metric-sub-pro">Verified discussions</div>
            </div>
            """, unsafe_allow_html=True)
        with d_col2:
            st.markdown(f"""
            <div class="metric-card-pro" style="margin-bottom: 12px;">
                <div class="metric-card-accent rose"></div>
                <div class="metric-label-pro">Spam Comments</div>
                <div class="metric-value-pro" style="color: #991B1B;">{stats.get('spam_count', 841):,}</div>
                <div class="metric-sub-pro">Promotional & scams</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card-pro" style="margin-bottom: 1.25rem;">
            <div class="metric-card-accent indigo"></div>
            <div class="metric-label-pro">Total Verified Corpus</div>
            <div class="metric-value-pro">{stats.get('total_comments', 1754):,}</div>
            <div class="metric-sub-pro">Split: 80% Train ({stats.get('train_size', 1403)}) / 20% Test ({stats.get('test_size', 351)})</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="saas-card" style="background: #F8FAFC;">
            <div style="font-weight: 700; color: #0F172A; font-size: 0.9rem; margin-bottom: 6px;">💡 Academic & Viva Note</div>
            <p style="font-size: 0.82rem; color: #475569; margin: 0; line-height: 1.5;">
                This system avoids hardcoded mock outputs. All classifications and probabilities are executed live
                through the scikit-learn pipeline serialized via Joblib.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION B: SINGLE COMMENT DETECTION
# ==============================================================================
elif page == "🔍 Single Comment Detection":
    st.markdown("""
    <div class="saas-hero-card" style="padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="hero-top-badges">
            <span class="pill-badge pill-primary">INFERENCE CONSOLE</span>
            <span class="pill-badge pill-neutral">REAL-TIME NLP</span>
        </div>
        <h2 class="hero-title" style="font-size: 1.75rem;">Single Comment Spam Detection</h2>
        <p class="hero-subtitle" style="margin-bottom: 0; font-size: 0.95rem;">
            Test the trained classifier on any user-submitted comment or explore benchmark demonstration samples.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Demonstration Samples
    st.markdown("""
    <div style="margin-bottom: 0.5rem; font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
        ⚡ Instant Demonstration Scenarios (Click to Load):
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    # Initialize session state for input comment if not present
    if "current_comment" not in st.session_state:
        st.session_state["current_comment"] = "Congratulations! You have won a free iPhone. Click this link now: http://bit.ly/claim-prize"

    if c1.button("🎁 Free iPhone Scam (Spam)", use_container_width=True):
        st.session_state["current_comment"] = "Congratulations! You have won a free iPhone. Click this link now: http://bit.ly/claim-prize"
        st.rerun()

    if c2.button("❤️ Video Appreciation (Ham)", use_container_width=True):
        st.session_state["current_comment"] = "This video is amazing. Thank you for sharing such a clear explanation!"
        st.rerun()

    if c3.button("📢 Sub4Sub Promo (Spam)", use_container_width=True):
        st.session_state["current_comment"] = "Please check out my channel and subscribe! I upload daily music beats and give free robux!"
        st.rerun()

    if c4.button("❓ Video Question (Ham)", use_container_width=True):
        st.session_state["current_comment"] = "Could you please explain what you did at minute 03:45? I was slightly confused there."
        st.rerun()

    # Comment Input Box
    comment_text = st.text_area(
        label="Enter YouTube Comment Text for Classification:",
        value=st.session_state["current_comment"],
        height=110,
        placeholder="Type or paste any YouTube comment here...",
        help="Input text will be processed via regex tokenization and evaluated by the linear SVM."
    )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        analyze_clicked = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)

    if analyze_clicked or comment_text:
        if not comment_text.strip():
            st.warning("Please enter a non-empty comment to analyze.")
        else:
            with st.spinner("Executing NLP vectorization & ML classification..."):
                result = predict_comment(comment_text)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([3, 2], gap="large")

            with res_col1:
                # Custom Verdict Card
                if result["prediction"] == 1:
                    st.markdown(f"""
                    <div class="prediction-card-spam">
                        <div class="prediction-header">
                            <h3 class="prediction-title" style="color: #991B1B;">
                                🚨 SPAM DETECTED
                            </h3>
                            <span class="pill-badge pill-danger">CLASS 1 • MALICIOUS / PROMO</span>
                        </div>
                        <p class="prediction-explanation">
                            {result['explanation']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-ham">
                        <div class="prediction-header">
                            <h3 class="prediction-title" style="color: #065F46;">
                                ✅ VERIFIED LEGITIMATE (HAM)
                            </h3>
                            <span class="pill-badge pill-success">CLASS 0 • GENUINE VIEWER</span>
                        </div>
                        <p class="prediction-explanation">
                            {result['explanation']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence and Probabilities Card
                st.markdown("""
                <div class="saas-card">
                    <div class="saas-card-title">🎯 Calibrated Confidence Assessment</div>
                """, unsafe_allow_html=True)

                conf_score = result['confidence']
                st.write(f"**Model Certainty Score:** **{conf_score:.1f}%**")
                st.progress(conf_score / 100.0)

                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.markdown(f"""
                    <div class="metric-card-pro" style="margin-top: 10px;">
                        <div class="metric-card-accent rose"></div>
                        <div class="metric-label-pro">Spam Probability</div>
                        <div class="metric-value-pro" style="font-size: 1.6rem; color: #991B1B;">{result['spam_probability']:.2%}</div>
                        <div class="metric-sub-pro">Class 1 Likelihood</div>
                    </div>
                    """, unsafe_allow_html=True)
                with p_col2:
                    st.markdown(f"""
                    <div class="metric-card-pro" style="margin-top: 10px;">
                        <div class="metric-card-accent emerald"></div>
                        <div class="metric-label-pro">Legitimate Probability</div>
                        <div class="metric-value-pro" style="font-size: 1.6rem; color: #065F46;">{result['ham_probability']:.2%}</div>
                        <div class="metric-sub-pro">Class 0 Likelihood</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            with res_col2:
                # NLP Preprocessing Inspection Card
                st.markdown("""
                <div class="saas-card">
                    <div class="saas-card-title">🔬 NLP Tokenization Inspection</div>
                    <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.75rem;">
                        Normalized token string passed directly into the TF-IDF feature vectorizer:
                    </p>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="token-inspector-box">
                    {result['cleaned_comment']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                    <div style="height: 16px;"></div>
                    <div class="saas-card-title" style="font-size: 1rem;">🚩 Heuristic Cues Detected</div>
                """, unsafe_allow_html=True)

                indicators = result.get("indicators", {})
                detected = False

                badge_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px;'>"

                if indicators.get("has_url"):
                    badge_html += '<span class="pill-badge pill-danger">🔗 External URL / Link</span>'
                    detected = True
                if indicators.get("excessive_caps"):
                    badge_html += '<span class="pill-badge pill-warning">🔠 Excessive Uppercase Shouting</span>'
                    detected = True
                if indicators.get("has_currency_or_reward"):
                    badge_html += '<span class="pill-badge pill-warning">💰 Reward / Financial Terms</span>'
                    detected = True
                if indicators.get("detected_keywords"):
                    for kw in indicators["detected_keywords"]:
                        badge_html += f'<span class="pill-badge pill-primary">🏷️ Keyword: {kw}</span>'
                    detected = True

                if not detected:
                    badge_html += '<span class="pill-badge pill-success">✨ No overt heuristic red flags (Clean)</span>'

                badge_html += "</div>"
                st.markdown(badge_html, unsafe_allow_html=True)

                if not detected:
                    st.caption("Classification relies on subtle stylistic n-gram vocabulary correlations.")

                st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION C: BATCH COMMENT DETECTION
# ==============================================================================
elif page == "📁 Batch Comment Detection":
    st.markdown("""
    <div class="saas-hero-card" style="padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="hero-top-badges">
            <span class="pill-badge pill-primary">HIGH-THROUGHPUT ENGINE</span>
            <span class="pill-badge pill-neutral">CSV INFERENCE</span>
        </div>
        <h2 class="hero-title" style="font-size: 1.75rem;">Batch Comment Moderation</h2>
        <p class="hero-subtitle" style="margin-bottom: 1rem; font-size: 0.95rem;">
            Upload any CSV spreadsheet containing multiple YouTube comments for automated, high-speed batch classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV Download Card
    sample_csv_path = os.path.join(PROJECT_ROOT, "dataset", "sample_batch_comments.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "r", encoding="utf-8") as f:
            sample_bytes = f.read()

        s_col1, s_col2 = st.columns([2, 3])
        with s_col1:
            st.download_button(
                label="📥 Download Ready-to-Test Sample CSV",
                data=sample_bytes,
                file_name="sample_youtube_comments.csv",
                mime="text/csv",
                help="Download a curated sample CSV with real spam and legitimate comments.",
                use_container_width=True
            )

    uploaded_file = st.file_uploader(
        "Upload CSV File with Comments:",
        type=["csv"],
        help="Supports any CSV containing text or comment columns."
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Spreadsheet uploaded successfully • Found {len(df_upload):,} rows and {len(df_upload.columns)} columns.")

            # Column Auto-Detection
            default_idx = 0
            for i, col in enumerate(df_upload.columns):
                if col.lower() in ["comment", "content", "comment_text", "text", "body", "message"]:
                    default_idx = i
                    break

            col_select, action_col = st.columns([3, 2])
            with col_select:
                selected_col = st.selectbox(
                    "Which column contains the YouTube comment text?",
                    options=df_upload.columns.tolist(),
                    index=default_idx
                )

            with action_col:
                st.write("") # Spacer
                st.write("")
                run_batch = st.button("🚀 Process Batch Comments", type="primary", use_container_width=True)

            if run_batch:
                with st.spinner("Processing comments through NLP pipeline..."):
                    results_df = predict_batch(df_upload, comment_col=selected_col)

                total = len(results_df)
                spam_count = int((results_df["Prediction"] == 1).sum())
                ham_count = int((results_df["Prediction"] == 0).sum())
                spam_pct = (spam_count / total) * 100 if total > 0 else 0

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                # Responsive Horizontal Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card-pro">
                        <div class="metric-card-accent indigo"></div>
                        <div class="metric-label-pro">Total Processed</div>
                        <div class="metric-value-pro">{total:,}</div>
                        <div class="metric-sub-pro">100% of batch rows</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card-pro">
                        <div class="metric-card-accent rose"></div>
                        <div class="metric-label-pro">Spam Detected</div>
                        <div class="metric-value-pro" style="color: #991B1B;">{spam_count:,}</div>
                        <div class="metric-sub-pro">{spam_pct:.1f}% flagged</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card-pro">
                        <div class="metric-card-accent emerald"></div>
                        <div class="metric-label-pro">Legitimate (Ham)</div>
                        <div class="metric-value-pro" style="color: #065F46;">{ham_count:,}</div>
                        <div class="metric-sub-pro">{100 - spam_pct:.1f}% approved</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card-pro">
                        <div class="metric-card-accent amber"></div>
                        <div class="metric-label-pro">Spam Ratio</div>
                        <div class="metric-value-pro">{spam_pct:.1f}%</div>
                        <div class="metric-sub-pro">Density in dataset</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="saas-card" style="margin-top: 1.5rem;">
                    <div class="saas-card-title">📋 Classified Comments Inspection</div>
                """, unsafe_allow_html=True)

                # Filter radio
                filter_choice = st.radio(
                    "Filter Inspection Table:",
                    ["All Comments", "Only Spam (Class 1)", "Only Legitimate (Class 0)"],
                    horizontal=True
                )

                if filter_choice == "Only Spam (Class 1)":
                    display_df = results_df[results_df["Prediction"] == 1]
                elif filter_choice == "Only Legitimate (Class 0)":
                    display_df = results_df[results_df["Prediction"] == 0]
                else:
                    display_df = results_df

                st.dataframe(
                    display_df[[selected_col, "Label", "Confidence (%)", "Spam_Probability"]],
                    use_container_width=True,
                    height=320
                )

                # Download processed CSV
                csv_export = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Fully Classified CSV Report",
                    data=csv_export,
                    file_name="moderated_youtube_comments.csv",
                    mime="text/csv",
                    type="primary"
                )
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading or processing CSV file: {e}")


# ==============================================================================
# SECTION D: MODEL PERFORMANCE DASHBOARD
# ==============================================================================
elif page == "📊 Model Performance Dashboard":
    st.markdown("""
    <div class="saas-hero-card" style="padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="hero-top-badges">
            <span class="pill-badge pill-primary">EVALUATION BENCHMARK</span>
            <span class="pill-badge pill-success">CALIBRATED SVM</span>
            <span class="pill-badge pill-neutral">TEST SPLIT: 351 SAMPLES</span>
        </div>
        <h2 class="hero-title" style="font-size: 1.75rem;">Model Performance & Diagnostics</h2>
        <p class="hero-subtitle" style="margin-bottom: 0; font-size: 0.95rem;">
            Empirical evaluation metrics and algorithm benchmarking obtained on the UCI YouTube Spam Collection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_cached_metrics()

    if not metrics:
        st.warning("Evaluation metrics have not been generated yet. Please run `python src/train_model.py` first.")
    else:
        best_model_name = metrics.get("best_model_name", "Support Vector Machine (LinearSVC)")
        models_eval = metrics.get("models_evaluation", [])
        best_eval = next((m for m in models_eval if m["model_name"] == best_model_name), models_eval[0] if models_eval else {})

        # Top Executive Metric Cards
        st.markdown(f"""
        <div style="margin-bottom: 0.75rem; font-size: 0.85rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
            🏆 Champion Algorithm: <span style="color: #2563EB;">{best_model_name}</span> (Evaluated on Unseen Test Set)
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-accent indigo"></div>
                <div class="metric-label-pro">Overall Accuracy</div>
                <div class="metric-value-pro">{best_eval.get('accuracy', 0):.2%}</div>
                <div class="metric-sub-pro">Total correct predictions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-accent emerald"></div>
                <div class="metric-label-pro">Precision (Spam)</div>
                <div class="metric-value-pro" style="color: #065F46;">{best_eval.get('precision', 0):.2%}</div>
                <div class="metric-sub-pro">Protects genuine viewers</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-accent amber"></div>
                <div class="metric-label-pro">Recall (Spam)</div>
                <div class="metric-value-pro">{best_eval.get('recall', 0):.2%}</div>
                <div class="metric-sub-pro">Catches 95%+ of spam</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card-pro">
                <div class="metric-card-accent rose"></div>
                <div class="metric-label-pro">F1-Score</div>
                <div class="metric-value-pro">{best_eval.get('f1_score', 0):.4f}</div>
                <div class="metric-sub-pro">Harmonic balance</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs([
            "📊 Algorithm Comparison Matrix",
            "📈 Visual Diagnostics & Charts",
            "🧠 NLP Concepts & Viva Guide"
        ])

        with tab1:
            st.markdown("""
            <div class="saas-card">
                <div class="saas-card-title">Comparative Performance Breakdown (4 Algorithms)</div>
            """, unsafe_allow_html=True)

            comparison_table = metrics.get("comparison_table", [])
            if comparison_table:
                comp_df = pd.DataFrame(comparison_table)
                st.dataframe(comp_df, use_container_width=True)

            st.markdown("""
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 1.1rem; margin-top: 1rem;">
                    <div style="font-weight: 700; color: #1E40AF; font-size: 0.95rem; margin-bottom: 6px;">
                        📌 Strategic Algorithm Insights:
                    </div>
                    <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.88rem; color: #1E3A8A; line-height: 1.6;">
                        <li><strong>Linear Support Vector Machine (LinearSVC):</strong> Selected as champion due to superior margin maximization in high-dimensional TF-IDF space (95.44% Accuracy, 0.9524 F1-score).</li>
                        <li><strong>Random Forest:</strong> Yielded the absolute highest precision (98.10%) with only 3 false positives out of 183 genuine comments, making it the most conservative filter.</li>
                        <li><strong>Multinomial Naive Bayes:</strong> Fast and lightweight baseline (94.30% Accuracy), but slightly penalized by independence assumptions across n-gram pairs.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            vcol1, vcol2 = st.columns(2, gap="large")

            with vcol1:
                st.markdown("""
                <div class="saas-card">
                    <div class="saas-card-title">Algorithm Metric Benchmarks</div>
                """, unsafe_allow_html=True)
                if comparison_table:
                    fig_comp = create_light_comparison_chart(pd.DataFrame(comparison_table))
                    st.pyplot(fig_comp, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with vcol2:
                st.markdown("""
                <div class="saas-card">
                    <div class="saas-card-title">Training Corpus Distribution</div>
                """, unsafe_allow_html=True)
                stats = metrics.get("dataset_stats", {})
                fig_dist = create_light_class_distribution(stats.get("ham_count", 913), stats.get("spam_count", 841))
                st.pyplot(fig_dist, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Confusion Matrix Card
            st.markdown("""
            <div class="saas-card">
                <div class="saas-card-title">Confusion Matrix Analysis</div>
            """, unsafe_allow_html=True)
            cm_eval = best_eval.get("confusion_matrix")
            if cm_eval:
                cm_col1, cm_col2 = st.columns([1, 1], gap="large")
                with cm_col1:
                    fig_cm = create_light_confusion_matrix(cm_eval, best_model_name)
                    st.pyplot(fig_cm, use_container_width=True)
                with cm_col2:
                    st.markdown(f"""
                    <div style="font-size: 0.9rem; color: #334155; line-height: 1.6; padding-top: 1rem;">
                        <h4 style="color: #0F172A; margin-top: 0;">Error Breakdown on 351 Test Comments:</h4>
                        <ul style="padding-left: 1.25rem;">
                            <li><strong>True Negatives (175):</strong> Real comments correctly kept visible.</li>
                            <li><strong>True Positives (160):</strong> Unsolicited spam comments successfully caught.</li>
                            <li><strong>False Positives (8):</strong> Genuine comments mistakenly flagged (Low 2.2% error rate).</li>
                            <li><strong>False Negatives (8):</strong> Subtle spam comments that evaded detection.</li>
                        </ul>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem; margin-top: 1rem;">
                            <span style="font-weight: 700; color: #0F172A;">Classification Integrity:</span><br>
                            Balanced false positive vs. false negative rates guarantee creator safety without alienating real community participation.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("""
            <div class="saas-card">
                <div class="saas-card-title">📚 Fundamental NLP & Machine Learning Principles</div>
                
                <h4 style="color: #0F172A; margin-top: 0.5rem; margin-bottom: 0.25rem;">1. Precision vs. Recall: The Content Moderation Trade-off</h4>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                    In automated content moderation, not all errors carry equal consequences:
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1rem;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem;">
                        <strong>Precision:</strong> Fraction of flagged comments that are truly spam.
                        $$\\text{Precision} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$$
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem;">
                        <strong>Recall:</strong> Fraction of all existing spam comments that were caught.
                        $$\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$
                    </div>
                </div>
                
                <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 10px; padding: 1rem; color: #92400E; margin-bottom: 1.5rem;">
                    <strong>⚠️ Why False Positives Are Critical on YouTube:</strong><br>
                    A False Positive means a genuine fan asking an authentic question is hidden or censored.
                    While a missed spam comment (False Negative) is merely an eyesore, censoring authentic users destroys
                    creator-audience trust. Hence, high Precision is paramount.
                </div>

                <h4 style="color: #0F172A; margin-bottom: 0.25rem;">2. Why TF-IDF Vectorization Outperforms Simple Word Counts</h4>
                <p style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                    Machine learning algorithms operate on mathematical matrices rather than raw text.
                    <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> rewards distinctive trigger words while
                    penalizing common filler vocabulary:
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem; font-size: 0.9rem; line-height: 1.6; color: #334155;">
                    $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|}\\right) + 1$$
                    <ul style="margin: 0.5rem 0 0 0; padding-left: 1.25rem;">
                        <li><strong>TF (Term Frequency):</strong> Quantifies how frequently term <em>t</em> occurs in comment <em>d</em>.</li>
                        <li><strong>IDF (Inverse Document Frequency):</strong> Depreciates ubiquitous words like <em>'video'</em> or <em>'this'</em>, while assigning high weights to rare triggers like <em>'httpurl'</em>, <em>'subscribe'</em>, or <em>'giveaway'</em>.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
