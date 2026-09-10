"""
threshold_sweep.py -- Finds the best MISMATCH_SIMILARITY_THRESHOLD for the
real sentence-transformers model, by testing several thresholds against
the ground-truth mismatches and reporting precision/recall/F1 for each.

Run this LOCALLY (needs the real model -- see prototype/README.md).

Run from the `experiments/` folder:
    python3 threshold_sweep.py
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join("..", "prototype"))
from sentence_transformers import SentenceTransformer, util  # noqa: E402

DATA_DIR = os.path.join("..", "synthetic_data")

HIGH_RATING_THRESHOLD = 7
NEGATIVE_BARRIER_KEYWORDS = [
    "relapse", "reluctance", "struggl", "disrupted", "conflict", "overwhelm",
]
RATING_DROP_THRESHOLD = 3
ROLLING_WINDOW = 3


def main():
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    sessions = sessions.sort_values(["goal_id", "session_date"]).reset_index(drop=True)

    print("Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Precompute similarity scores once for all sessions with real barriers
    print("Computing embeddings for all sessions...")
    sims = []
    for _, row in sessions.iterrows():
        barriers = str(row["barriers"])
        text = str(row["self_reported_text"])
        if barriers.strip().lower() in ("no barriers this session", "nan", ""):
            sims.append(1.0)
            continue
        emb = model.encode([text, barriers], convert_to_tensor=True)
        sims.append(float(util.cos_sim(emb[0], emb[1])))
    sessions["sim"] = sims

    # Precompute rating-drop and rating/keyword flags once (threshold-independent)
    rating_drop_flags = []
    for goal_id, group in sessions.groupby("goal_id"):
        ratings = group["self_reported_rating"].tolist()
        for i in range(len(group)):
            history = ratings[:i][-ROLLING_WINDOW:]
            if history:
                avg = sum(history) / len(history)
                rating_drop_flags.append(avg - ratings[i] >= RATING_DROP_THRESHOLD)
            else:
                rating_drop_flags.append(False)
    sessions["rating_drop_flag"] = rating_drop_flags

    sessions["rating_keyword_flag"] = sessions.apply(
        lambda r: r["self_reported_rating"] >= HIGH_RATING_THRESHOLD
        and any(k in str(r["barriers"]).lower() for k in NEGATIVE_BARRIER_KEYWORDS),
        axis=1,
    )

    gt = sessions["flagged_mismatch_ground_truth"].astype(bool)

    print(f"\n{'Threshold':<12}{'TP':<6}{'FP':<6}{'FN':<6}{'TN':<6}{'Precision':<12}{'Recall':<10}{'F1':<8}")
    print("-" * 65)

    best_f1, best_thresh = 0, None
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (
            (sessions["sim"] < thresh)
            | sessions["rating_keyword_flag"]
            | sessions["rating_drop_flag"]
        )
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        tn = int((~pred & ~gt).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"{thresh:<12}{tp:<6}{fp:<6}{fn:<6}{tn:<6}{precision:<12.2f}{recall:<10.2f}{f1:<8.2f}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"\nBest threshold: {best_thresh} (F1={best_f1:.2f})")
    print(f"\nUpdate MISMATCH_SIMILARITY_THRESHOLD in prototype/mismatch_detector.py to {best_thresh}")


if __name__ == "__main__":
    main()