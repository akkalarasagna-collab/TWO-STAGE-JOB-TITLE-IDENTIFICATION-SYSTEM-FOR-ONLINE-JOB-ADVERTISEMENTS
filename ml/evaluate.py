# ml/evaluate.py — Load saved models and print detailed evaluation metrics

import os
import sys

# Allow imports from project root when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ml.preprocess import preprocess_text


def evaluate():
    # ── 1. Resolve paths ──────────────────────────────────────────────────
    base_dir   = os.path.join(os.path.dirname(__file__), '..')
    data_path  = os.path.normpath(os.path.join(base_dir, 'data', 'job_data.csv'))
    model_dir  = os.path.normpath(os.path.join(base_dir, 'models'))
    vec_path   = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(model_dir, 'logistic_model.pkl')

    # ── 2. Check files exist ──────────────────────────────────────────────
    for path, label in [(data_path, 'Dataset'), (vec_path, 'Vectorizer'), (model_path, 'Model')]:
        if not os.path.exists(path):
            print(f"[evaluate.py] ERROR: {label} not found at {path}")
            if label == 'Dataset':
                print("  → Run: python ml/dataset.py")
            else:
                print("  → Run: python ml/train.py")
            sys.exit(1)

    # ── 3. Load data ──────────────────────────────────────────────────────
    print(f"[evaluate.py] Loading dataset from {data_path} ...")
    df = pd.read_csv(data_path)
    df['cleaned'] = df['description'].apply(preprocess_text)

    # Re-create the same 80/20 split used during training
    _, X_test, _, y_test = train_test_split(
        df['cleaned'], df['title'],
        test_size=0.20,
        random_state=42,
        stratify=df['title']
    )

    # ── 4. Load models ────────────────────────────────────────────────────
    print("[evaluate.py] Loading saved models ...")
    vectorizer = joblib.load(vec_path)
    model      = joblib.load(model_path)

    # ── 5. Vectorize test set ─────────────────────────────────────────────
    X_test_vec = vectorizer.transform(X_test)

    # ── 6. Predict ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)

    # ── 7. Metrics ────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  Test Accuracy: {acc * 100:.2f}%")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    labels = sorted(df['title'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Pretty-print the confusion matrix
    col_width = max(len(l) for l in labels) + 2
    header = ' ' * col_width + ''.join(l[:6].ljust(8) for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        row_str = label.ljust(col_width) + ''.join(str(v).ljust(8) for v in row)
        print(row_str)

    print("\n[evaluate.py] Evaluation complete.")


if __name__ == '__main__':
    evaluate()
