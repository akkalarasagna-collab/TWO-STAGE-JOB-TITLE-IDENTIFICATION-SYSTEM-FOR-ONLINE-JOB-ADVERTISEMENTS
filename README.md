# Two-Stage Job Title Identification System

**AI-powered web application that identifies job titles from online job advertisement text.**

Built with Python · Flask · Scikit-learn · NLTK · Hugging Face Transformers · SQLite

---

## What It Does

Paste any job description — from LinkedIn, Indeed, Naukri, or anywhere — and the system returns:

- **Predicted job title** (e.g. Flutter Developer, Machine Learning Engineer)
- **Job category** (e.g. Software Development, Artificial Intelligence)
- **Confidence score** with animated progress bar
- **Top-3 alternative title suggestions**
- **Technical skills** detected (50 skills: Python, Docker, React, PyTorch…)
- **Soft skills** detected (15 skills: Leadership, Teamwork, Problem Solving…)
- **Prediction history** stored in SQLite with analytics dashboard

---

## How It Works — Two-Stage Pipeline

```
Job Description (raw text)
        │
        ▼
┌─────────────────────────────┐
│  STAGE 1: Pattern Matching  │  ← Regex + keywords, ~50 patterns
│  app/stage1_matcher.py      │    Handles: "flutter", "wordpress",
└──────────────┬──────────────┘    "terraform", "spring boot", etc.
         Found?│
    Yes ◄──────┤──────► No
     │                   │
  100% / 90%             ▼
  confidence   ┌─────────────────────────┐
               │  STAGE 2: BART NLI      │  ← facebook/bart-large-mnli
               │  bert_classifier.py     │    Zero-shot classification
               │  15 candidate labels    │    Margin-based confidence
               └─────────────────────────┘
                          │
               ┌──────────▼──────────────┐
               │  Skill Extraction       │  ← 65 keyword patterns
               │  app/skill_extractor.py │
               └──────────┬──────────────┘
                          │
               ┌──────────▼──────────────┐
               │  SQLite Persistence     │  ← history.db
               │  database.py            │
               └─────────────────────────┘
```

**Stage 1** runs in microseconds using pre-compiled regex patterns for technology keywords (`\bflutter\b`, `\bwoocommerce\b`, `\bterraform\b`) and explicit title phrases (`\bflutter\s+developer\b`). Returns 90% confidence for keyword signals, 100% for explicit phrases.

**Stage 2** uses `facebook/bart-large-mnli` — a 400M-parameter transformer model — for zero-shot NLI classification. No retraining required to add new titles. Uses a margin-based confidence formula that combines absolute score strength with decisiveness over the next-best candidate.

---

## Job Titles Supported

| Category | Titles |
|---|---|
| Software Development | Software Engineer, Backend Developer, Full Stack Developer, Node.js Developer, Flutter Developer, iOS Developer, Java Developer, JavaScript Developer, PHP Developer, Django Developer, Wordpress Developer |
| Artificial Intelligence | Machine Learning Engineer |
| DevOps & Cloud | DevOps Engineer, Network Administrator, Database Administrator |

Stage 1 additionally detects: Frontend Developer, Mobile Developer, QA Engineer, Data Scientist, Data Analyst, Data Engineer, Cloud Engineer, Cybersecurity Analyst, Business Analyst, Product Manager, Project Manager, Scrum Master, and more.

---

## Project Structure

```
job_title_identifier/
│
├── run.py                      # Flask entry point — python run.py
├── config.py                   # DEBUG, SECRET_KEY, MODEL_PATH
├── database.py                 # SQLite CRUD (history.db)
├── bert_classifier.py          # Stage 2: BART zero-shot NLI
├── prepare_dataset.py          # Merge + clean training CSVs
├── requirements.txt            # Python dependencies
│
├── app/
│   ├── __init__.py             # Flask app factory
│   ├── routes.py               # URL handlers (/predict, /history, /api/history)
│   ├── predictor.py            # CATEGORY_MAP + legacy TF-IDF Predictor class
│   ├── stage1_matcher.py       # Stage 1 regex/keyword matching (3-pass)
│   ├── skill_extractor.py      # 50 tech + 15 soft skill patterns
│   └── utils.py                # sanitize, truncate, format helpers
│
├── ml/
│   ├── train.py                # Train TF-IDF + Logistic Regression
│   ├── preprocess.py           # NLTK text cleaning (lowercase/stopwords/lemmatize)
│   ├── evaluate.py             # Classification report + confusion matrix
│   ├── dataset.py              # Legacy synthetic dataset generator
│   └── __init__.py
│
├── models/
│   ├── tfidf_vectorizer.pkl    # Trained TF-IDF vectoriser
│   └── logistic_model.pkl      # Trained Logistic Regression model
│
├── data/
│   └── job_data.csv            # Legacy synthetic training data
│
├── templates/
│   ├── index.html              # Main prediction page (Bootstrap 5)
│   └── history.html            # Prediction history dashboard
│
└── static/
    ├── css/style.css           # Dark navy theme, animations
    └── js/main.js              # AJAX, DOM rendering, confidence bar
```

---

## Quick Start

### 1. Clone / download the project

```bash
cd job_title_identifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Prepare training data and train the model

```bash
# Place job_title_des.csv in the project root, then:
python prepare_dataset.py       # generates merged_train_temp.csv
python ml/train.py              # trains TF-IDF + LR, saves .pkl files
```

### 5. Start the app

```bash
python run.py
```

Open your browser at **http://localhost:5000**

---

## Training Pipeline

The TF-IDF + Logistic Regression model is trained on `merged_train_temp.csv` — a cleaned, deduplicated dataset sourced from `job_title_des.csv` plus hand-written discriminative samples.

### Model parameters

**TF-IDF Vectoriser:**
```python
TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=10000,
    min_df=2,
    sublinear_tf=True,
    stop_words='english'
)
```

**Logistic Regression:**
```python
LogisticRegression(
    C=5.0,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)
```

### Training results (2,297 samples, 15 titles)

```
Training Accuracy : 99.18%
Test Accuracy     : 84.35%
Macro avg F1      : 0.85
```

**Per-class F1 scores:**

| Title | F1 |
|---|---|
| iOS Developer | 0.97 |
| Network Administrator | 0.97 |
| Machine Learning Engineer | 0.97 |
| Database Administrator | 0.93 |
| Django Developer | 0.93 |
| Wordpress Developer | 0.95 |
| DevOps Engineer | 0.92 |
| Node.js Developer | 0.88 |
| PHP Developer | 0.85 |
| JavaScript Developer | 0.85 |
| Flutter Developer | 0.88 |
| Full Stack Developer | 0.75 |
| Java Developer | 0.67 |
| Software Engineer | 0.63 |
| Backend Developer | 0.58 |

---

## API Endpoints

| Method | URL | Description |
|---|---|---|
| `GET` | `/` | Main prediction page |
| `POST` | `/predict` | Predict job title from description |
| `GET` | `/history` | Prediction history dashboard |
| `GET` | `/api/history` | JSON: last 20 predictions + stats |
| `DELETE` | `/api/history` | Clear all history |

### POST /predict — Request

```json
{
  "description": "We are looking for a Flutter developer..."
}
```

### POST /predict — Response

```json
{
  "stage": 1,
  "detection_stage": "Stage 1 – Keyword & Pattern Matching",
  "job_title": "Flutter Developer",
  "job_category": "Software Development",
  "confidence": 90.0,
  "alternatives": [],
  "technical_skills": ["Flutter", "Dart", "Firebase", "REST API"],
  "soft_skills": ["Problem Solving", "Teamwork"]
}
```

---

## Confidence Score Logic

**Stage 1** returns:
- `100%` — explicit title phrase found (`"flutter developer"`, `"devops engineer"`)
- `90%` — strong single-signal keyword found (`"flutter"`, `"woocommerce"`, `"terraform"`)

**Stage 2 (BART)** uses a margin-based formula:

```python
# Component 1: absolute strength (0.15–0.65 NLI range)
abs_component = (top_score - 0.15) / (0.65 - 0.15)

# Component 2: margin over next-best candidate
margin_component = (top_score - second_score) / 0.15

# Combined: 60% absolute + 40% decisiveness
confidence = max(15.0, min(98.0, (0.6 * abs + 0.4 * margin) * 100))
```

**Confidence tiers in the UI:**
- 🟢 Green — ≥ 75% (High)
- 🟡 Orange — 45–74% (Medium)
- 🔴 Red — < 45% (Low)

---

## Smoke Test (no Flask required)

```bash
python _smoke_test.py
```

Tests 5 real-world descriptions directly against the prediction functions. All 5 currently pass:

```
Test 1 — Flutter Developer      Stage 1   90%   PASS ✓
Test 2 — Wordpress Developer    Stage 1   90%   PASS ✓
Test 3 — Full Stack Developer   Stage 1  100%   PASS ✓
Test 4 — Backend Developer      Stage 1  100%   PASS ✓
Test 5 — Software Engineer      Stage 1  100%   PASS ✓
```

---

## Adding New Job Titles

No retraining needed for Stage 2. Three files to edit:

**1. `bert_classifier.py` — add to `CANDIDATE_LABELS`:**
```python
CANDIDATE_LABELS = [
    ...
    "Your New Title",   # just append here
]
```

**2. `bert_classifier.py` — add to `LABEL_DESCRIPTIONS`:**
```python
LABEL_DESCRIPTIONS = {
    ...
    "Your New Title": "keywords describing this role",
}
```

**3. `app/predictor.py` — add to `CATEGORY_MAP`:**
```python
CATEGORY_MAP = {
    ...
    "Your New Title": "Category Name",
}
```

**4. (Optional) `app/stage1_matcher.py` — add a regex pattern for explicit matching:**
```python
REGEX_PATTERNS = [
    ...
    (r'\byour\s+new\s+title\b', 'Your New Title'),
]
```

---

## Requirements

```
flask
scikit-learn
pandas
nltk
sentence-transformers       # installs torch + transformers automatically
```

**Python:** 3.9+ (tested on 3.14)
**BART model:** `facebook/bart-large-mnli` (~1.5 GB, cached in `~/.cache/huggingface/hub/`)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask 3.x |
| ML classifier (Stage 2) | facebook/bart-large-mnli (Hugging Face) |
| ML classifier (legacy/training) | Scikit-learn TF-IDF + Logistic Regression |
| Text preprocessing | NLTK (stopwords, WordNetLemmatizer) |
| Database | SQLite via Python `sqlite3` |
| Frontend | Bootstrap 5, Font Awesome 6, Vanilla JS |
| Serialisation | Joblib (.pkl files) |

---

## Academic Context

Built as a B.Tech Mini Project demonstrating:
- NLP text preprocessing pipeline
- TF-IDF feature engineering with n-grams
- Multi-class Logistic Regression classification
- Zero-shot NLI classification with transformer models
- Flask REST API design with Blueprint architecture
- SQLite persistence and analytics
- Responsive single-page web application (AJAX, no page reloads)

---

*Two Stage Job Title Identification System — Academic Project*
