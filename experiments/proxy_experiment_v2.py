"""
Proxy experiment v2 -- adds the rating-drop check (rolling avg vs current
rating) on top of the original TF-IDF + keyword checks, to measure its
effect on recall as requested in review feedback.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sessions = pd.read_csv("../synthetic_data/sessions.csv")
sessions = sessions.sort_values(["goal_id", "session_date"]).reset_index(drop=True)

HIGH_RATING = 7
NEG_KEYWORDS = ["relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm"]
SIM_THRESHOLD = 0.0
RATING_DROP_THRESHOLD = 3
ROLLING_WINDOW = 3

def sim_score(text, barriers):
    if str(barriers).strip().lower() in ("no barriers this session", "nan", ""):
        return 1.0
    try:
        v = TfidfVectorizer().fit_transform([str(text), str(barriers)])
        return cosine_similarity(v[0], v[1])[0][0]
    except ValueError:
        return 1.0

results = []
for goal_id, group in sessions.groupby("goal_id"):
    ratings = group["self_reported_rating"].tolist()
    for i, (_, row) in enumerate(group.iterrows()):
        reasons = []
        sim = sim_score(row["self_reported_text"], row["barriers"])
        if sim < SIM_THRESHOLD:
            reasons.append("semantic")
        barriers_lower = str(row["barriers"]).lower()
        if row["self_reported_rating"] >= HIGH_RATING and any(k in barriers_lower for k in NEG_KEYWORDS):
            reasons.append("rating_keyword")
        history = ratings[:i][-ROLLING_WINDOW:]
        if history:
            avg = sum(history) / len(history)
            if avg - row["self_reported_rating"] >= RATING_DROP_THRESHOLD:
                reasons.append("rating_drop")
        results.append({"idx": row.name, "flagged": len(reasons) > 0, "reasons": reasons})

pred_df = pd.DataFrame(results).set_index("idx")
sessions["predicted_flag_v2"] = pred_df["flagged"]
sessions["reasons_v2"] = pred_df["reasons"]

gt = sessions["flagged_mismatch_ground_truth"].astype(bool)
pred = sessions["predicted_flag_v2"].astype(bool)

tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
precision = tp/(tp+fp) if tp+fp else 0
recall = tp/(tp+fn) if tp+fn else 0
f1 = 2*precision*recall/(precision+recall) if precision+recall else 0

print("=== BEFORE (v1, no rating-drop check) ===")
print("TP=12 FP=54 FN=24 TN=203  Precision=0.18 Recall=0.33 F1=0.24")
print()
print("=== AFTER (v2, with rating-drop check) ===")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}  Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")

# How many of the newly-caught cases were specifically rating_drop catches?
rating_drop_catches = sum(1 for r in results if "rating_drop" in r["reasons"] and sessions.loc[r["idx"], "flagged_mismatch_ground_truth"])
print(f"\nTrue mismatches caught specifically via the new rating-drop check: {rating_drop_catches}")