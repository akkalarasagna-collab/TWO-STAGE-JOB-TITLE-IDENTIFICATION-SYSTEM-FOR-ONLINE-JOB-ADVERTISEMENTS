# bert_classifier.py — Zero-Shot NLI classifier for Stage 2 prediction
#
# Model: facebook/bart-large-mnli
#   - De-facto standard for zero-shot classification via NLI
#   - Fully cached locally (~1.5 GB) — no download needed after first run
#   - Uses the Hugging Face zero-shot-classification pipeline
#   - Returns properly normalised probabilities across all candidate labels
#
# No training data needed. To add new job titles, append to CANDIDATE_LABELS.

import re
import numpy as np
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"

# ── Candidate job title labels ────────────────────────────────────────────────
CANDIDATE_LABELS = [
    "Software Engineer",
    "Backend Developer",
    "Full Stack Developer",
    "Node.js Developer",
    "Flutter Developer",
    "iOS Developer",
    "Java Developer",
    "JavaScript Developer",
    "PHP Developer",
    "Django Developer",
    "Wordpress Developer",
    "DevOps Engineer",
    "Network Administrator",
    "Database Administrator",
    "Machine Learning Engineer",
]

# ── Label descriptions — used to guide the NLI hypothesis ────────────────────
# Each value is a short keyword-rich description that helps BART-MNLI understand
# the role better when framing the NLI hypothesis.
LABEL_DESCRIPTIONS = {
    "Software Engineer":       "software engineer develop build applications systems code architecture programming",
    "Backend Developer":       "backend developer server side API REST database microservices Python Java Node",
    "Full Stack Developer":    "full stack developer frontend backend React Node.js database end to end web",
    "Node.js Developer":       "Node.js developer backend JavaScript server Express REST API MongoDB",
    "Flutter Developer":       "Flutter developer mobile Dart cross-platform iOS Android UI widgets Firebase",
    "iOS Developer":           "iOS developer Swift Objective-C Xcode Apple iPhone iPad mobile app",
    "Java Developer":          "Java developer Spring Boot Hibernate REST API Maven backend enterprise",
    "JavaScript Developer":    "JavaScript developer ES6 React Angular Vue Node frontend web development",
    "PHP Developer":           "PHP developer Laravel CodeIgniter MySQL backend web application API",
    "Django Developer":        "Django developer Python REST framework PostgreSQL backend web API",
    "Wordpress Developer":     "WordPress developer PHP themes plugins WooCommerce CMS website",
    "DevOps Engineer":         "DevOps engineer CI CD pipelines Kubernetes Docker Terraform AWS infrastructure",
    "Network Administrator":   "network administrator LAN WAN firewall routers switches Cisco infrastructure",
    "Database Administrator":  "database administrator DBA SQL Server Oracle MySQL PostgreSQL optimization",
    "Machine Learning Engineer": "machine learning engineer MLOps TensorFlow PyTorch deep learning model training",
}

# Module-level pipeline cache — loaded once at startup, reused for all requests
_clf_pipeline = None

# ── Boilerplate label patterns — lines starting with these are stripped ───────
_BOILERPLATE_RE = re.compile(
    r'^(?:'
    r'Schedule'
    r'|Job\s+Type'
    r'|Pay'
    r'|Salary'
    r'|Benefits'
    r'|Ability\s+to\s+commute(?:/relocate)?'
    r'|Experience'
    r'|Education'
    r'|Location'
    r'|Application\s+Question'
    r'|License(?:/Certification)?'
    r')\s*:',
    re.IGNORECASE,
)

# Lines following an "Experience:" header that contain these words are also removed
_EXPERIENCE_DETAIL_RE = re.compile(
    r'\b(years?|preferred|required)\b',
    re.IGNORECASE,
)


def clean_description_for_stage2(text: str) -> str:
    """
    Strip common job-portal boilerplate lines from *text* before BERT encoding.

    Real postings from Indeed / Naukri / LinkedIn often append sections like:
        "Schedule: Day shift"
        "Job Type: Full-time"
        "Ability to commute/relocate: Chandigarh..."
        "Experience: total work: 5 years (Preferred)"
        "Benefits: Health + Dental"

    These lines dilute the semantic signal the BART model needs to identify
    the job title.

    Steps:
      1. Split into lines.
      2. Remove any line that starts with a known boilerplate label.
      3. Remove lines immediately following an "Experience:" header that
         contain "years", "Preferred", or "Required".
      4. Collapse blank lines and normalise whitespace.
      5. Fallback: if the cleaned text is shorter than 30 characters,
         return the original uncleaned text (nothing useful was left).

    Returns:
        str — cleaned text, or original text if cleaning stripped too much.
    """
    if not text:
        return text

    lines = text.splitlines()
    cleaned_lines = []
    skip_next = False   # flag: skip experience-detail lines after "Experience:"

    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            skip_next = False
            continue

        # If the previous line was "Experience:", skip detail lines like
        # "total work: 5 years (Preferred)"
        if skip_next:
            if _EXPERIENCE_DETAIL_RE.search(stripped):
                continue
            else:
                skip_next = False  # non-detail line — stop skipping

        # Check if this line starts with a boilerplate label
        if _BOILERPLATE_RE.match(stripped):
            # If it is an "Experience:" line, set flag for the next line
            if re.match(r'^Experience\s*:', stripped, re.IGNORECASE):
                skip_next = True
            continue

        cleaned_lines.append(stripped)

    # Join, collapse multiple spaces, strip edges
    cleaned = ' '.join(cleaned_lines)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()

    # Fallback: if cleaning left almost nothing, use the original
    if len(cleaned) < 30:
        return text.strip()

    return cleaned


def load_bert_model():
    """
    Load the zero-shot classification pipeline exactly once.

    Model: facebook/bart-large-mnli
    Loaded from local Hugging Face cache (~1.5 GB).
    Subsequent calls return the already-loaded pipeline immediately.
    """
    global _clf_pipeline
    if _clf_pipeline is None:
        print(f"[bert_classifier] Loading {MODEL_NAME} from local cache …")
        _clf_pipeline = pipeline(
            task="zero-shot-classification",
            model=MODEL_NAME,
        )
        print("[bert_classifier] Model loaded and ready.")
    return _clf_pipeline


def _margin_confidence(scores_array: np.ndarray, idx: int) -> float:
    """
    Compute a human-readable confidence percentage for the candidate at *idx*
    using a margin-based formula that combines:

      Component 1 (60 %) — absolute score strength:
        How strong is this label's raw score relative to the expected range
        for good NLI entailment matches (0.15 … 0.65)?

      Component 2 (40 %) — decisiveness margin:
        How much does this label beat the next-ranked label below it?
        A margin of 0.15 or more is considered fully decisive.

    Returns a float in [15.0, 98.0].
    """
    sorted_indices = np.argsort(scores_array)[::-1].tolist()

    this_score  = float(scores_array[idx])

    # Find the score of the item ranked one position below this one
    rank = sorted_indices.index(idx)
    if rank + 1 < len(sorted_indices):
        next_score = float(scores_array[sorted_indices[rank + 1]])
    else:
        next_score = 0.0

    # Component 1: absolute strength (typical NLI entailment range 0.15–0.65)
    abs_component = (this_score - 0.15) / (0.65 - 0.15)
    abs_component = max(0.0, min(1.0, abs_component))

    # Component 2: decisiveness margin over next competitor
    margin = this_score - next_score
    margin_component = margin / 0.15       # margin ≥ 0.15 → fully decisive
    margin_component = max(0.0, min(1.0, margin_component))

    combined   = (0.6 * abs_component) + (0.4 * margin_component)
    confidence = round(max(15.0, min(98.0, combined * 100)), 1)
    return confidence


def predict_with_bert(description: str) -> dict:
    """
    Classify *description* against all CANDIDATE_LABELS using zero-shot NLI
    and return a structured prediction result.

    Pipeline:
      1. Clean boilerplate (clean_description_for_stage2)
      2. Truncate to 512 characters
      3. Run BART-MNLI zero-shot inference with hypothesis template
      4. Apply margin-based confidence formula to top prediction
      5. Build top-3 alternatives with their own margin-based scores
      6. Return result dict

    Parameters
    ----------
    description : str
        Raw job description text (may contain portal boilerplate).

    Returns
    -------
    dict with keys:
        predicted_title  (str)   — top-ranked job title
        confidence       (float) — margin-based percentage [15.0, 98.0]
        alternatives     (list)  — top-3 runner-up dicts:
                                   {'title': str, 'confidence': float}
    """
    # ── Step 1: Strip boilerplate noise ───────────────────────────────────────
    description = clean_description_for_stage2(description)

    # ── Step 2: Truncate to stay within BART token limits ─────────────────────
    text = description[:512]

    # ── Step 3: Zero-shot NLI inference ───────────────────────────────────────
    clf = load_bert_model()

    # Use enriched label descriptions when available — this gives BART-MNLI
    # much richer signal to discriminate between similar developer roles.
    enriched_labels = [
        LABEL_DESCRIPTIONS.get(lbl, lbl) for lbl in CANDIDATE_LABELS
    ]

    result = clf(
        text,
        candidate_labels=enriched_labels,
        # Hypothesis template frames the task as a proper NLI entailment question.
        hypothesis_template="This job description is for a {}.",
        multi_label=False,   # softmax normalisation — scores sum to ≈ 1.0
    )

    # result['labels'] and result['scores'] are sorted highest → lowest
    # Labels returned are the enriched descriptions — map back to canonical names
    enriched_to_canonical = dict(zip(enriched_labels, CANDIDATE_LABELS))
    labels     = [enriched_to_canonical.get(lbl, lbl) for lbl in result["labels"]]
    raw_scores = result["scores"]    # list[float], sum ≈ 1.0

    # Convert to numpy array indexed in CANDIDATE_LABELS order
    # (the pipeline returns them sorted, so we need to re-align indices)
    label_to_score = dict(zip(labels, raw_scores))
    scores_array   = np.array(
        [label_to_score.get(lbl, 0.0) for lbl in CANDIDATE_LABELS],
        dtype=np.float64,
    )

    # ── Step 4: Top prediction with margin-based confidence ───────────────────
    sorted_indices  = np.argsort(scores_array)[::-1].tolist()
    top_idx         = sorted_indices[0]
    predicted_title = CANDIDATE_LABELS[top_idx]
    confidence      = _margin_confidence(scores_array, top_idx)

    # ── Step 5: Top-3 alternatives with their own margin scores ───────────────
    alternatives = []
    for i in sorted_indices[1:4]:
        alt_conf = _margin_confidence(scores_array, i)
        alternatives.append({
            "title":      CANDIDATE_LABELS[i],
            "confidence": alt_conf,
        })

    return {
        "predicted_title": predicted_title,
        "confidence":      confidence,
        "alternatives":    alternatives,
    }
