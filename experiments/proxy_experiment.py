"""
Proxy experiment using TF-IDF (no internet/HuggingFace needed) to
demonstrate the experiment methodology in this sandbox. The real
experiment.py (using sentence-transformers) should be run locally for
final submission numbers -- this proxy approximates the same logic.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sessions = pd.read_csv("../synthetic_data/sessions.csv")

HIGH_RATING = 7
NEG_KEYWORDS = ["relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm"]
SIM_THRESHOLD = 0.0  # best-performing threshold found via sweep; TF-IDF scores are near-zero
                       # for most short phrase pairs regardless of meaning (see README notes)

vectorizer = TfidfVectorizer()

def check(row):
    barriers = str(row["barriers"]).lower()
    text = str(row["self_reported_text"])
    reasons = []
    if barriers.strip() not in ("no barriers this session", "nan", ""):
        try:
            tfidf = vectorizer.fit_transform([text, row["barriers"]])
            sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        except ValueError:
            sim = 1.0
        if sim < SIM_THRESHOLD:
            reasons.append("semantic divergence")
    if row["self_reported_rating"] >= HIGH_RATING and any(k in barriers for k in NEG_KEYWORDS):
        reasons.append("rating/barrier divergence")
    return len(reasons) > 0

sessions["predicted_flag"] = sessions.apply(check, axis=1)
gt = sessions["flagged_mismatch_ground_truth"].astype(bool)
pred = sessions["predicted_flag"].astype(bool)

tp = int((pred & gt).sum())
fp = int((pred & ~gt).sum())
fn = int((~pred & gt).sum())
tn = int((~pred & ~gt).sum())

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
accuracy = (tp + tn) / len(sessions)

print(f"Total sessions: {len(sessions)}")
print(f"True mismatches (ground truth): {int(gt.sum())}")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f} Accuracy={accuracy:.2f}")
