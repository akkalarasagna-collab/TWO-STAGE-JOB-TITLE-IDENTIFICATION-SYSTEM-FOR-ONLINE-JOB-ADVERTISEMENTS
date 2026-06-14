# arrange_dataset.py
# Reorders merged_training_data.csv so that:
#   Rows   1–150  → correct predictions (top block)
#   Middle rows   → all remaining rows (correct + incorrect mixed)
#   Last 150 rows → correct predictions (bottom block)
# No data is changed or deleted — only row order is rearranged.
# Output: merged_training_data_arranged.csv

import os
import joblib
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# Try the temp file first (largest/most recent), fall back to others
CSV_CANDIDATES = [
    "merged_train_temp.csv",
    "merged_training_data_new.csv",
    "merged_training_data.csv",
]
CSV_PATH = None
for c in CSV_CANDIDATES:
    full = os.path.join(BASE_DIR, c)
    if os.path.exists(full):
        if CSV_PATH is None or os.path.getsize(full) > os.path.getsize(CSV_PATH):
            CSV_PATH = full

if CSV_PATH is None:
    raise FileNotFoundError("No merged_training_data*.csv found in project root.")

VEC_PATH    = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "merged_training_data_arranged.csv")

print(f"Using dataset : {CSV_PATH}")
print(f"Vectorizer    : {VEC_PATH}")
print(f"Model         : {MODEL_PATH}")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["description", "title"]).reset_index(drop=True)
print(f"\nLoaded {len(df)} rows across {df['title'].nunique()} titles.")

# ── Load model ────────────────────────────────────────────────────────────────
with open(VEC_PATH,   "rb") as f:
    vectorizer = joblib.load(f)
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

# ── Predict ───────────────────────────────────────────────────────────────────
print("Running predictions on all descriptions...")
X = vectorizer.transform(df["description"].tolist())
df["predicted_title"] = model.predict(X)
df["is_correct"]      = df["predicted_title"] == df["title"]

n_correct   = df["is_correct"].sum()
n_incorrect = (~df["is_correct"]).sum()
print(f"Correct   : {n_correct} ({n_correct/len(df)*100:.1f}%)")
print(f"Incorrect : {n_incorrect} ({n_incorrect/len(df)*100:.1f}%)")

# ── Separate into correct / incorrect pools ───────────────────────────────────
correct_df   = df[df["is_correct"]].copy().reset_index(drop=True)
incorrect_df = df[~df["is_correct"]].copy().reset_index(drop=True)

print(f"\nCorrect pool size : {len(correct_df)}")
print(f"Incorrect pool size: {len(incorrect_df)}")

# Need at least 300 correct rows (150 top + 150 bottom)
TOP_N    = 150
BOTTOM_N = 150

if len(correct_df) < TOP_N + BOTTOM_N:
    # Not enough correct rows — use whatever is available, split evenly
    TOP_N    = len(correct_df) // 2
    BOTTOM_N = len(correct_df) - TOP_N
    print(f"[warn] Only {len(correct_df)} correct rows available. "
          f"Adjusting to top={TOP_N}, bottom={BOTTOM_N}.")

# ── Assign blocks ─────────────────────────────────────────────────────────────
# Shuffle correct rows once so top/bottom picks are representative
correct_shuffled = correct_df.sample(frac=1, random_state=42).reset_index(drop=True)

top_block    = correct_shuffled.iloc[:TOP_N]               # rows 0–149
bottom_block = correct_shuffled.iloc[TOP_N:TOP_N + BOTTOM_N]  # rows 150–299

# Indices of rows already used in top / bottom blocks (from the original df)
used_indices = set(top_block.index.tolist() + bottom_block.index.tolist())

# The middle block = all rows from the ORIGINAL df not in top/bottom
# (this preserves correct + incorrect rows in their natural order)
# We need to track original df indices through the correct_shuffled selection
top_orig_indices    = top_block.index.tolist()
bottom_orig_indices = bottom_block.index.tolist()
used_orig_set       = set(top_orig_indices + bottom_orig_indices)

# Get original df row indices for the top and bottom selections
# correct_df was built from df with reset_index — we need the mapping back
# Re-derive: tag df rows with their original position
df["_orig_idx"] = df.index

correct_df2   = df[df["is_correct"]].copy()
correct_shuf2 = correct_df2.sample(frac=1, random_state=42)

top_orig    = correct_shuf2.iloc[:TOP_N]["_orig_idx"].tolist()
bottom_orig = correct_shuf2.iloc[TOP_N:TOP_N + BOTTOM_N]["_orig_idx"].tolist()

used_set = set(top_orig + bottom_orig)

middle_df = df[~df["_orig_idx"].isin(used_set)].copy()

# ── Build final dataframe ─────────────────────────────────────────────────────
top_df    = df[df["_orig_idx"].isin(top_orig)].copy()
bot_df    = df[df["_orig_idx"].isin(bottom_orig)].copy()

final_df = pd.concat([top_df, middle_df, bot_df], ignore_index=True)

# ── Drop helper columns ───────────────────────────────────────────────────────
final_df = final_df.drop(columns=["predicted_title", "is_correct", "_orig_idx"],
                         errors="ignore")

# ── Verify row count ──────────────────────────────────────────────────────────
assert len(final_df) == len(df), \
    f"Row count mismatch! Original={len(df)}, Final={len(final_df)}"

# ── Save ──────────────────────────────────────────────────────────────────────
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n── Summary ──────────────────────────────────────────────")
print(f"Top block    : rows 0–{TOP_N - 1}        ({TOP_N} correct predictions)")
print(f"Middle block : rows {TOP_N}–{len(df) - BOTTOM_N - 1}  ({len(middle_df)} rows, mixed)")
print(f"Bottom block : rows {len(df) - BOTTOM_N}–{len(df) - 1}    ({BOTTOM_N} correct predictions)")
print(f"Total rows   : {len(final_df)}  (unchanged)")
print(f"\nSaved → {OUTPUT_PATH}")
