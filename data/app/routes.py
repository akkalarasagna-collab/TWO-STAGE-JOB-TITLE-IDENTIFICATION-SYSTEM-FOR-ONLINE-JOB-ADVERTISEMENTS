# app/routes.py — URL routes and request handling

from flask import Blueprint, render_template, request, jsonify
from .stage1_matcher import Stage1Matcher
from .skill_extractor import SkillExtractor
from .predictor import CATEGORY_MAP   # category map still used for both stages
from bert_classifier import predict_with_bert
import database

main = Blueprint('main', __name__)

# Instantiate helpers once at import time (cheap)
stage1          = Stage1Matcher()
skill_extractor = SkillExtractor()


# ── Main page ─────────────────────────────────────────────────────────────────

@main.route('/', methods=['GET'])
def index():
    """Render the main single-page UI."""
    return render_template('index.html')


# ── Prediction endpoint ───────────────────────────────────────────────────────

@main.route('/predict', methods=['POST'])
def predict():
    """
    Predict job title from a submitted job description.

    Flow:
      1. Validate input.
      2. Stage 1 — keyword / regex matching (instant, no ML).
      3. Stage 2 — Zero-Shot DistilBERT classifier (if Stage 1 misses).
      4. Extract technical and soft skills.
      5. Persist result to SQLite history.
      6. Return JSON response.
    """
    try:
        # ── Input validation ──────────────────────────────────────────────
        data        = request.get_json(silent=True) or {}
        description = data.get('description', '').strip()

        if not description:
            return jsonify({'error': 'Job description is required.'}), 400
        if len(description) < 20:
            return jsonify({'error': 'Job description is too short. Please provide more detail.'}), 400

        # ── Stage 1: Keyword / Regex matching ─────────────────────────────
        # match() now returns (title, confidence) tuple
        stage1_title, stage1_conf = stage1.match(description)

        if stage1_title:
            job_category = CATEGORY_MAP.get(stage1_title, 'General')
            technical_skills, soft_skills = skill_extractor.extract(description)

            response = {
                'stage':            1,
                'detection_stage':  'Stage 1 – Keyword & Pattern Matching',
                'job_title':        stage1_title,
                'job_category':     job_category,
                'confidence':       float(stage1_conf),
                'alternatives':     [],
                'technical_skills': technical_skills,
                'soft_skills':      soft_skills,
            }
            database.save_prediction({**response, 'description': description})
            return jsonify(response), 200

        # ── Stage 2: Zero-Shot DistilBERT ──────────────────────────────────
        bert_result = predict_with_bert(description)

        job_title    = bert_result['predicted_title']
        confidence   = bert_result['confidence']

        # Map BERT alternatives (use 'title' key to match frontend expectation)
        alternatives = [
            {'title': alt['title'], 'probability': alt['confidence']}
            for alt in bert_result['alternatives']
        ]

        job_category = CATEGORY_MAP.get(job_title, 'General')
        technical_skills, soft_skills = skill_extractor.extract(description)

        response = {
            'stage':            2,
            'detection_stage':  'Stage 2 – Zero-Shot DistilBERT (Hugging Face)',
            'job_title':        job_title,
            'job_category':     job_category,
            'confidence':       confidence,
            'alternatives':     alternatives,
            'technical_skills': technical_skills,
            'soft_skills':      soft_skills,
        }
        database.save_prediction({**response, 'description': description})
        return jsonify(response), 200

    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {str(exc)}'}), 500


# ── History page ──────────────────────────────────────────────────────────────

@main.route('/history', methods=['GET'])
def history():
    """Render the prediction history page."""
    return render_template('history.html')


# ── History API ───────────────────────────────────────────────────────────────

@main.route('/api/history', methods=['GET'])
def api_history():
    """Return the last 20 predictions and summary stats as JSON."""
    try:
        records = database.get_history(limit=20)
        stats   = database.get_stats()
        return jsonify({'records': records, 'stats': stats}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@main.route('/api/history', methods=['DELETE'])
def api_history_clear():
    """Delete all prediction history records."""
    try:
        database.clear_history()
        return jsonify({'success': True}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
