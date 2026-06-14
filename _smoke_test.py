"""
Direct smoke test — calls prediction functions without Flask or HTTP.
Run from project root: python _smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.stage1_matcher import Stage1Matcher
from bert_classifier    import predict_with_bert, load_bert_model
from app.predictor      import CATEGORY_MAP

TESTS = [
    {
        "label":    "Test 1 — Flutter Developer",
        "desc":     "We are looking for a Flutter Developer to build cross-platform mobile "
                    "apps for iOS and Android using Dart and Flutter SDK. Experience with "
                    "Firebase, REST APIs, and BLoC state management required.",
        "expected": "Flutter Developer",
    },
    {
        "label":    "Test 2 — Wordpress Developer",
        "desc":     "We need a WordPress Developer to build and customize WordPress websites. "
                    "Strong PHP skills, custom theme and plugin development, WooCommerce setup, "
                    "and MySQL database management required.",
        "expected": "Wordpress Developer",
    },
    {
        "label":    "Test 3 — Full Stack Developer",
        "desc":     "Full Stack Developer needed with React and Node.js experience. Build React "
                    "frontend with Redux, Node.js Express backend REST APIs, MongoDB database, "
                    "and AWS deployment.",
        "expected": "Full Stack Developer",
    },
    {
        "label":    "Test 4 — Backend Developer",
        "desc":     "Backend developer to build REST APIs using Python FastAPI. Work with "
                    "PostgreSQL database, Redis caching, Docker deployment, JWT authentication, "
                    "and AWS Lambda. No frontend work required.",
        "expected": "Backend Developer",
    },
    {
        "label":    "Test 5 — Software Engineer",
        "desc":     "Software Engineer to build scalable backend systems in Python. Design "
                    "microservices architecture, REST APIs, PostgreSQL optimization, Docker "
                    "containerization, and Kubernetes deployment on AWS.",
        "expected": "Software Engineer",
    },
]

stage1 = Stage1Matcher()

print("Loading BERT model from local cache...")
load_bert_model()
print("Model ready.\n")

print("=" * 65)
print("  Direct Prediction Smoke Test (no Flask / no HTTP)")
print("=" * 65)

all_passed = True

for t in TESTS:
    desc = t["desc"]

    s1_title, s1_conf = stage1.match(desc)

    if s1_title:
        stage        = 1
        predicted    = s1_title
        confidence   = float(s1_conf)
        alternatives = []
    else:
        stage       = 2
        result      = predict_with_bert(desc)
        predicted   = result["predicted_title"]
        confidence  = result["confidence"]
        alternatives = result["alternatives"]

    category = CATEGORY_MAP.get(predicted, "General")
    passed   = (predicted == t["expected"])
    if not passed:
        all_passed = False

    status = "PASS ✓" if passed else "FAIL ✗"

    print(f"\n{t['label']}")
    print(f"  Stage      : Stage {stage}")
    print(f"  Predicted  : {predicted}")
    print(f"  Expected   : {t['expected']}")
    print(f"  Category   : {category}")
    print(f"  Confidence : {confidence}%")
    if alternatives:
        print(f"  Alternatives:")
        for a in alternatives[:3]:
            pct = a.get("confidence") or a.get("probability") or 0
            print(f"    • {a.get('title','?')}: {pct}%")
    print(f"  Result     : {status}")

print()
print("=" * 65)
print("ALL TESTS PASSED ✓" if all_passed else "SOME TESTS FAILED ✗")
print("=" * 65)
