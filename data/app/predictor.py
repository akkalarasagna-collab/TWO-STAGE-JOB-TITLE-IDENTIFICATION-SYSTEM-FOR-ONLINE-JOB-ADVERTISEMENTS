# app/predictor.py — Stage 2: ML-based job title prediction

import os
import joblib
import numpy as np
from ml.preprocess import preprocess_text

# ── Job Category Mapping ─────────────────────────────────────────────────────
CATEGORY_MAP = {
    "Software Engineer":       "Software Development",
    "Backend Developer":       "Software Development",
    "Full Stack Developer":    "Software Development",
    "Node.js Developer":       "Software Development",
    "Flutter Developer":       "Software Development",
    "iOS Developer":           "Software Development",
    "Java Developer":          "Software Development",
    "JavaScript Developer":    "Software Development",
    "PHP Developer":           "Software Development",
    "Django Developer":        "Software Development",
    "Wordpress Developer":     "Software Development",
    "DevOps Engineer":         "DevOps & Cloud",
    "Network Administrator":   "DevOps & Cloud",
    "Database Administrator":  "DevOps & Cloud",
    "Machine Learning Engineer": "Artificial Intelligence",
    # Keep broader fallbacks for any titles that appear via Stage 1 matching
    "Frontend Developer":      "Software Development",
    "Mobile Developer":        "Software Development",
    "Software Architect":      "Software Development",
    "Systems Architect":       "Software Development",
    "QA Engineer":             "Software Development",
    "UI/UX Designer":          "Software Development",
    "Data Scientist":          "Data Science",
    "Data Analyst":            "Data Science",
    "Data Engineer":           "Data Science",
    "AI Engineer":             "Artificial Intelligence",
    "AI/ML Engineer":          "Artificial Intelligence",
    "NLP Engineer":            "Artificial Intelligence",
    "Computer Vision Engineer":"Artificial Intelligence",
    "Cloud Engineer":          "DevOps & Cloud",
    "Site Reliability Engineer":"DevOps & Cloud",
    "Network Engineer":        "DevOps & Cloud",
    "Security Engineer":       "Cyber Security",
    "Cybersecurity Analyst":   "Cyber Security",
    "Cyber Security Analyst":  "Cyber Security",
    "Business Analyst":        "Business & Management",
    "Product Manager":         "Business & Management",
    "Project Manager":         "Business & Management",
    "Scrum Master":            "Business & Management",
}


class Predictor:
    """
    Loads the trained TF-IDF vectorizer and Logistic Regression model once
    at startup and exposes a predict() method.
    """

    def __init__(self, model_path: str):
        """
        Load models from *model_path* directory.

        Raises FileNotFoundError if model files are missing (run train.py first).
        """
        vectorizer_file = os.path.join(model_path, 'tfidf_vectorizer.pkl')
        model_file      = os.path.join(model_path, 'logistic_model.pkl')

        if not os.path.exists(vectorizer_file):
            raise FileNotFoundError(
                f'TF-IDF vectorizer not found at {vectorizer_file}. '
                'Run "python ml/train.py" first.'
            )
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f'Logistic Regression model not found at {model_file}. '
                'Run "python ml/train.py" first.'
            )

        self.vectorizer = joblib.load(vectorizer_file)
        self.model      = joblib.load(model_file)
        self.classes    = list(self.model.classes_)

    def predict(self, description: str) -> dict:
        """
        Preprocess *description*, vectorize, and predict job title.

        Returns a dict:
          {
            'title':        str,
            'category':     str,
            'confidence':   float,   # percentage, e.g. 87.3
            'alternatives': list[dict]  # top 3 non-winning classes
          }
        """
        # Step 1 — Clean the text
        cleaned = preprocess_text(description)

        # Step 2 — Vectorize
        features = self.vectorizer.transform([cleaned])

        # Step 3 — Predict with probabilities
        probabilities = self.model.predict_proba(features)[0]

        # Step 4 — Rank classes by probability (descending)
        ranked_indices = np.argsort(probabilities)[::-1]

        best_idx  = ranked_indices[0]
        job_title = self.classes[best_idx]

        # ── Scaled confidence ─────────────────────────────────────────────
        # Raw softmax probabilities from Logistic Regression are "compressed"
        # toward 1/n (random chance) when classes are similar.
        # Scaling stretches the range (1/n … 1) onto (0 … 100) so that a
        # prediction that is just barely above chance reads as ~0 % and a
        # near-certain prediction reads close to 99 %.
        n        = len(self.classes)
        raw_conf = float(probabilities[best_idx])
        scaled   = (raw_conf - 1 / n) / (1 - 1 / n) * 100
        confidence = max(10.0, min(99.0, round(scaled, 1)))

        # Step 5 — Build top-3 alternatives (excluding the winner), also scaled
        alternatives = []
        for idx in ranked_indices[1:4]:
            alt_title    = self.classes[idx]
            alt_raw      = float(probabilities[idx])
            alt_scaled   = (alt_raw - 1 / n) / (1 - 1 / n) * 100
            alt_prob     = max(0.0, round(alt_scaled, 1))
            alternatives.append({'title': alt_title, 'probability': alt_prob})

        # Step 6 — Resolve category
        category = CATEGORY_MAP.get(job_title, 'General')

        return {
            'title':        job_title,
            'category':     category,
            'confidence':   confidence,
            'alternatives': alternatives,
        }
