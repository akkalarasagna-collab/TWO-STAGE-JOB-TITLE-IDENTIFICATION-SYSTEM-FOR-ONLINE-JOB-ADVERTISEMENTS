# ml/train.py — Train TF-IDF + Logistic Regression on merged_training_data.csv
#
# Run from the project root: python ml/train.py

import os
import sys

# Allow imports from the project root when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ml.preprocess import preprocess_text


def train():
    # ── 1. Resolve paths ──────────────────────────────────────────────────
    base_dir  = os.path.join(os.path.dirname(__file__), '..')
    # Try candidate filenames in order — pick the largest (most up-to-date)
    candidates = [
        "merged_train_temp.csv",
        "merged_training_data_new.csv",
        "merged_training_data.csv",
    ]
    data_path = None
    for candidate in candidates:
        full = os.path.normpath(os.path.join(base_dir, candidate))
        if os.path.exists(full):
            if data_path is None or os.path.getsize(full) > os.path.getsize(data_path):
                data_path = full
    if data_path is None:
        print("[train.py] ERROR: No training CSV found. Run prepare_dataset.py first.")
        sys.exit(1)
    # Fall back to the _new file if the main one is still locked by Excel
    new_path  = os.path.normpath(os.path.join(base_dir, 'merged_training_data_new.csv'))
    if not os.path.exists(data_path) and os.path.exists(new_path):
        data_path = new_path
    elif os.path.exists(new_path):
        # Prefer the _new file if it is larger (has the extra samples)
        if os.path.getsize(new_path) > os.path.getsize(data_path):
            data_path = new_path
    model_dir = os.path.normpath(os.path.join(base_dir, 'models'))
    vec_path  = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    lr_path   = os.path.join(model_dir, 'logistic_model.pkl')

    os.makedirs(model_dir, exist_ok=True)

    # ── 2. Load dataset ───────────────────────────────────────────────────
    if not os.path.exists(data_path):
        print(f"[train.py] ERROR: {data_path} not found.")
        print("[train.py] Run 'python prepare_dataset.py' first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['title', 'description'])

    print(f"[train.py] Loaded {len(df)} rows across {df['title'].nunique()} titles.")
    print(f"[train.py] Distribution:\n{df['title'].value_counts().to_string()}\n")

    descriptions = df['description'].tolist()
    titles       = df['title'].tolist()

    # ── 3. Preprocess ─────────────────────────────────────────────────────
    print("[train.py] Preprocessing text ...")
    cleaned = [preprocess_text(d) for d in descriptions]

    # ── 4. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, titles,
        test_size=0.2,
        random_state=42,
        stratify=titles,
    )
    print(f"[train.py] Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── 5. TF-IDF ─────────────────────────────────────────────────────────
    print("[train.py] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
        stop_words='english',
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"[train.py] Vocabulary size: {len(vectorizer.vocabulary_)}\n")

    # ── 6. Logistic Regression ────────────────────────────────────────────
    print("[train.py] Training Logistic Regression ...")
    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    test_acc  = accuracy_score(y_test,  model.predict(X_test_vec))

    print(f"\n{'='*60}")
    print(f"  Training Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test Accuracy     : {test_acc  * 100:.2f}%")
    print(f"{'='*60}\n")
    print("Classification Report (Test Set):")
    print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))

    # ── 8. Save ───────────────────────────────────────────────────────────
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model,      lr_path)
    print(f"[train.py] Saved vectorizer → {vec_path}")
    print(f"[train.py] Saved model      → {lr_path}")
    print("[train.py] Done.")


if __name__ == '__main__':
    train()
