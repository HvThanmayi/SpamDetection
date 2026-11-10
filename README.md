# YouTube Comment Spam Detection Using Machine Learning

An end-to-end, functional Natural Language Processing (NLP) and Machine Learning prototype developed for undergraduate Computer Science and Engineering students as part of an **IBM Skill Lab Programme**.

This system automatically analyses YouTube comments and accurately classifies them into **SPAM (1)** or **HAM / NOT SPAM (0)** using Scikit-Learn, TF-IDF Vectorization, and an interactive Streamlit web interface.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [Technologies & Libraries Used](#-technologies--libraries-used)
- [Dataset Details](#-dataset-details)
- [Installation & Environment Setup](#-installation--environment-setup)
- [How to Train the Machine Learning Models](#-how-to-train-the-machine-learning-models)
- [How to Run the Streamlit Web Application](#-how-to-run-the-streamlit-web-application)
- [Actual Model Evaluation Results](#-actual-model-evaluation-results)
- [Demonstration Guide (Sample Test Comments)](#-demonstration-guide-sample-test-comments)
- [Beginner-Friendly Technical Explanations](#-beginner-friendly-technical-explanations)
- [Academic Viva / Project Review Q&A](#-academic-viva--project-review-qa)
- [Limitations & Future Enhancements](#-limitations--future-enhancements)

---

## 🎯 Project Overview

User-generated video sharing platforms like YouTube receive millions of comments daily. A substantial portion consists of:
- Unwanted self-promotional advertisements (e.g., *"Sub4Sub"*, *"Check my channel"*)
- Phishing links & fraudulent malware redirects
- Impersonators, cryptocurrency schemes, and fake prize giveaways

### Classification Target:
- **SPAM (`Class 1`):** Promotional comments, external phishing/shortened URLs, prize promises, duplicate spam bot messages.
- **NOT SPAM / HAM (`Class 0`):** Legitimate viewer opinions, appreciative feedback, genuine questions, timestamps, and conversational community remarks.

---

## 📂 Project Architecture & Directory Structure

```text
SpamDetector/
│
├── dataset/
│   ├── Youtube01-Psy.csv                 # Raw UCI dataset (Psy video comments)
│   ├── Youtube02-KatyPerry.csv           # Raw UCI dataset (Katy Perry video comments)
│   ├── Youtube03-LMFAO.csv               # Raw UCI dataset (LMFAO video comments)
│   ├── Youtube04-Eminem.csv              # Raw UCI dataset (Eminem video comments)
│   ├── Youtube05-Shakira.csv             # Raw UCI dataset (Shakira video comments)
│   ├── youtube_comments.csv              # Combined, deduplicated master dataset
│   └── sample_batch_comments.csv         # Sample test comments for batch evaluation demo
│
├── models/
│   ├── spam_detection_model.joblib       # Best trained ML pipeline (Joblib binary)
│   ├── evaluation_metrics.json           # Actual test metrics & model comparison records
│   ├── class_distribution.png            # Visual plot of dataset class balance
│   ├── model_comparison_chart.png        # Bar chart comparing all 4 ML models
│   ├── cm_support_vector_machine_linearsvc.png # Confusion matrix heatmap for SVM
│   ├── cm_logistic_regression.png        # Confusion matrix heatmap for Logistic Reg
│   ├── cm_multinomial_naive_bayes.png    # Confusion matrix heatmap for Naive Bayes
│   └── cm_random_forest_classifier.png   # Confusion matrix heatmap for Random Forest
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── preprocessing.py                  # Text cleaning, URL tokenizing, stopword handling
│   ├── evaluation.py                     # Performance calculation & Matplotlib/Seaborn plots
│   ├── train_model.py                    # Multi-model training, stratified split & model export
│   └── predict.py                        # Single and batch inference with confidence scoring
│
├── app.py                                # Interactive Streamlit Web Application
├── requirements.txt                      # Project dependency specification
├── .gitignore                            # Git exclusion rules
└── README.md                             # Comprehensive project documentation
```

### Purpose of Every Major File:
1. `src/preprocessing.py`: Implements NLP cleaning functions (`clean_text`, `extract_spam_cues`). Converts text to lowercase, handles HTML entities, replaces URLs with `httpurl`, strips punctuation, and protects critical spam keywords from stopword removal.
2. `src/train_model.py`: Automates dataset loading, deduplication, stratified train-test splitting (80/20), pipeline creation with `TfidfVectorizer`, training of 4 algorithms, evaluation on unseen test data, and serialization of the best model.
3. `src/evaluation.py`: Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and generates publication-quality visualization charts using Matplotlib and Seaborn.
4. `src/predict.py`: Provides persistent, cached model loading using Joblib. Formulates predictions for individual comments or batch DataFrames with calibrated probability confidence.
5. `app.py`: The user-facing Streamlit application featuring four complete sections: Home Overview, Single Comment Detection, Batch CSV Processing, and an interactive Model Performance Dashboard.

---

## 🛠️ Technologies & Libraries Used

| Technology | Purpose |
|---|---|
| **Python 3.12** | Core programming language |
| **NumPy** | High-performance numerical operations and array indexing |
| **Pandas** | Tabular data manipulation, CSV ingestion, and deduplication |
| **Scikit-learn** | Pipeline, TF-IDF Vectorizer, and classification algorithms |
| **Joblib** | Model serialization and disk persistence |
| **Matplotlib & Seaborn** | Statistical evaluation charts and confusion matrix heatmaps |
| **Streamlit** | Modern, interactive web user interface |

---

## 📊 Dataset Details

The system is trained on the publicly available **UCI Machine Learning Repository: YouTube Spam Collection Dataset**.
- **Source:** [UCI YouTube Spam Collection](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection)
- **Original Files:** 5 CSV files corresponding to comments on top music videos (Psy, Katy Perry, LMFAO, Eminem, Shakira).
- **Total Raw Records:** 1,956 comments.
- **After Deduplication:** 1,760 unique comments.
- **Class Breakdown:**
  - Legitimate (Ham, 0): **919 comments (52.2%)**
  - Spam (Spam, 1): **841 comments (47.8%)**

---

## 🚀 Installation & Environment Setup

### 1. Clone or Open the Repository
Open a terminal in the project directory:
```bash
cd c:\Users\thanm\OneDrive\Documents\SpamDetector
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ How to Train the Machine Learning Models

To re-run the training pipeline from scratch, execute:
```bash
python src/train_model.py
```

### What this command does:
1. Loads the 5 UCI artist CSV files from `dataset/`.
2. Cleans duplicates and empty comments, saving `dataset/youtube_comments.csv`.
3. Preprocesses text with `clean_text`.
4. Performs an 80/20 stratified split (`random_state=42`).
5. Fits a `TfidfVectorizer` (unigrams + bigrams, max 4,000 features).
6. Trains 4 candidate models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (`LinearSVC` with probability calibration)
   - Random Forest Classifier
7. Evaluates each model on the unseen test set (351 comments).
8. Exports the winning pipeline to `models/spam_detection_model.joblib`.
9. Writes evaluation metrics to `models/evaluation_metrics.json` and generates chart figures in `models/`.

---

## 💻 How to Run the Streamlit Web Application

Launch the Streamlit web application with:
```bash
streamlit run app.py
```
Or using the virtual environment python directly:
```bash
.\.venv\Scripts\streamlit.exe run app.py
```

Once running, open your web browser at:
```
http://localhost:8501
```

### Application Features:
- **🏠 Home & Overview:** Introduction, project objective, system architecture, and dataset summary.
- **🔍 Single Comment Detection:** Interactive text box with 4 one-click sample demonstration buttons, instant SPAM / NOT SPAM banner, confidence meter, normalized text inspection, and detected spam indicators.
- **📁 Batch Comment Detection:** Upload any CSV file, select the comment column, run batch inference, view instant statistics (Spam count, Ham count, Spam rate), filter results, and export the annotated CSV.
- **📊 Model Performance Dashboard:** Displays genuine test set metrics, comparative algorithm rankings, confusion matrix heatmaps, and plain-English educational explanations of ML concepts.

---

## 📈 Actual Model Evaluation Results

> [!IMPORTANT]
> The evaluation metrics below are **actual results** obtained on the 351 unseen test comments of the UCI YouTube Spam Collection. No results are fabricated.

| Algorithm | Accuracy | Precision | Recall | F1-Score | False Positives (Ham → Spam) | False Negatives (Spam → Ham) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (LinearSVC)** 🏆 | **95.44%** | **95.24%** | **95.24%** | **0.9524** | **8** | **8** |
| **Logistic Regression** | 95.44% | 96.34% | 94.05% | 0.9518 | 6 | 10 |
| **Random Forest Classifier** | 95.44% | 98.10% | 92.26% | 0.9509 | 3 | 13 |
| **Multinomial Naive Bayes** | 94.30% | 96.25% | 91.67% | 0.9390 | 6 | 14 |

### Metric Breakdown:
- **Winning Model:** Calibrated Linear Support Vector Machine (`LinearSVC`). It achieves a balanced **95.24% Precision** and **95.24% Recall** with an overall accuracy of **95.44%**.
- **Lowest False Positives:** Random Forest had only **3 False Positives** out of 183 legitimate test comments, delivering **98.10% Precision**.
- **Probability Calibration:** The winning SVM is wrapped in `CalibratedClassifierCV`, enabling calibrated `predict_proba()` output for accurate confidence percentages.

---

## 🧪 Demonstration Guide (Sample Test Comments)

Use these test comments during your project viva or lab review:

### Test Case 1: Phishing & Reward Scam
- **Input:** `"Congratulations! You have won a free iPhone. Click this link now: http://bit.ly/claim-prize"`
- **Expected Result:** `🚨 SPAM DETECTED` (Confidence: ~99%)
- **Why:** Contains URL pattern, reward term `"won a free iphone"`, and promotional imperative `"click this link"`.

### Test Case 2: Genuine Appreciation
- **Input:** `"This video is amazing. Thank you for sharing such a clear explanation!"`
- **Expected Result:** `✅ NOT SPAM (LEGITIMATE)` (Confidence: ~70%)
- **Why:** Exhibits positive sentiment and genuine viewer appreciation without promotional cues.

### Test Case 3: Channel Promotion (Sub4Sub)
- **Input:** `"Please check out my channel and subscribe! I upload daily music beats and give free robux!"`
- **Expected Result:** `🚨 SPAM DETECTED` (Confidence: 100%)
- **Why:** Matches high-weight spam n-grams: `"check out"`, `"my channel"`, `"subscribe"`, `"free"`.

### Test Case 4: Video Timestamp & Technical Question
- **Input:** `"Could you please explain what you did at minute 03:45? I was slightly confused there."`
- **Expected Result:** `✅ NOT SPAM (LEGITIMATE)` (Confidence: ~94%)
- **Why:** Normal conversational sentence structure typical of human viewers asking content-specific questions.

---

## 🧠 Beginner-Friendly Technical Explanations

### 1. What Does the Complete System Do?
It acts as a digital bouncer for YouTube comment sections. When a comment arrives, it cleans the text, converts it into numbers, and lets an algorithm decide if it's genuine viewer feedback or junk promotion.

### 2. What Happens When a User Enters a Comment?
1. The raw string goes to `clean_text()` in `src/preprocessing.py`.
2. Special characters and noise are stripped; links are standardized to `httpurl`.
3. The cleaned string enters the fitted `Pipeline`.
4. The `TfidfVectorizer` converts it into a sparse numerical vector.
5. The trained `LinearSVC` calculates the decision score and probability.
6. The Streamlit UI displays the badge, probability, and detected indicators.

### 3. Why is NLP Required?
Computers only understand numbers (`0` and `1`). They cannot read words like `"awesome"` or `"free money"`. NLP provides the bridge: it handles casing, removes grammar noise, standardizes links, and isolates informative terms.

### 4. Why TF-IDF (Term Frequency-Inverse Document Frequency)?
- **Term Frequency (TF):** If a comment repeats the word `"subscribe"` 3 times, that word is important inside that comment.
- **Inverse Document Frequency (IDF):** If the word `"video"` appears in every single comment, it has zero discriminative power. But if `"giveaway"` or `"httpurl"` appears almost exclusively in spam comments, IDF gives it a huge positive weight!

### 5. How Each ML Algorithm Works at a Basic Level:
- **Multinomial Naive Bayes:** Uses Bayes' Theorem to calculate the conditional probability of spam given the observed words, assuming each word is conditionally independent.
- **Logistic Regression:** Fits a linear decision boundary using the sigmoid function to map feature weights directly into a probability between 0 and 1.
- **Support Vector Machine (LinearSVC):** Finds the optimal hyperplane that separates spam from legitimate comments with the maximum geometric margin.
- **Random Forest:** An ensemble of decision trees that each vote on the classification, minimizing overfitting through bagging.

### 6. Why Pay Attention to False Positives?
- **False Positive (FP):** The model flags a genuine viewer's comment as spam.
- **Consequence:** An innocent fan's comment is deleted or hidden. The creator loses engagement and the viewer feels censored.
- **Conclusion:** While high recall is desirable, high **Precision** is critical in real-world moderation systems to protect legitimate community interaction.

---

## 🎓 Academic Viva / Project Review Q&A

**Q1: Why did you use TF-IDF instead of simple Bag of Words (CountVectorizer)?**
> *Answer:* CountVectorizer only counts raw frequency, meaning common words like "video" or "song" would dominate the feature space. TF-IDF penalizes universally frequent words through Inverse Document Frequency and rewards discriminative keywords that uniquely indicate spam.

**Q2: How did you avoid data leakage between training and testing?**
> *Answer:* We used Scikit-learn's `Pipeline`. The `TfidfVectorizer` was fitted solely on `X_train`. The test set `X_test` was only transformed using the already-fitted vocabulary, guaranteeing the model had zero prior knowledge of the test distribution.

**Q3: How does the model recognize URLs without overfitting to specific websites?**
> *Answer:* In `src/preprocessing.py`, our regex parser replaces specific web addresses with a generic semantic token `httpurl`. The model learns that *the presence of a link* is a spam signal, rather than memorizing individual domains.

**Q4: Why was LinearSVC selected over Naive Bayes?**
> *Answer:* High-dimensional text spaces (thousands of n-gram features) are linearly separable. LinearSVC finds the maximum margin hyperplane and achieved a higher F1-score (0.9524 vs 0.9390) and fewer false negatives.

---

## 🔮 Limitations & Future Enhancements

1. **Standalone Prototype Scope:** This system operates as a standalone classifier and is not hooked directly into live YouTube moderation APIs.
2. **Multilingual Comments:** The current prototype is optimized for English YouTube comments. Future versions can incorporate multilingual models (e.g., XLM-RoBERTa).
3. **Deep Learning & Transformers:** Future semesters can explore fine-tuning pre-trained transformers like `DistilBERT` or `BERTweet` for contextual semantic nuances.

---

## 👨‍💻 Student Project Team

- **Programme:** IBM Skill Lab Programme
- **Domain:** Artificial Intelligence & Machine Learning (CSE)
- **Academic Year:** 2024–2026
