"""
experiment.py -- Measurable comparison: Prototype vs Baseline

Run this LOCALLY (needs the real sentence-transformers model -- see
prototype/README.md for setup). It evaluates the mismatch-detection
prototype against the ground-truth mismatches that were deliberately
injected into the synthetic data generator, and contrasts that with what
the attendance-only baseline can (not) tell us.

Run from the `experiments/` folder:
    python3 experiment.py

Requires: pandas, sentence-transformers (see prototype/mismatch_detector.py)
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join("..", "prototype"))
from mismatch_detector import MismatchDetector  # noqa: E402

DATA_DIR = os.path.join("..", "synthetic_data")


def main():
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    detector = MismatchDetector(use_ollama=False)  # explanations not needed for metrics

    # Sort within each goal by date so rating history is chronological
    sessions = sessions.sort_values(["goal_id", "session_date"]).reset_index(drop=True)

    predictions = []
    for goal_id, group in sessions.groupby("goal_id"):
        ratings = group["self_reported_rating"].tolist()
        for i, (_, row) in enumerate(group.iterrows()):
            history = ratings[:i]
            result = detector.check_session(row.to_dict(), rating_history=history)
            predictions.append((row.name, result.is_flagged, result.severity))

    pred_map = {idx: (flag, sev) for idx, flag, sev in predictions}
    sessions["predicted_flag"] = sessions.index.map(lambda i: pred_map[i][0])
    sessions["severity"] = sessions.index.map(lambda i: pred_map[i][1])
    ground_truth = sessions["flagged_mismatch_ground_truth"].astype(bool)
    predicted = sessions["predicted_flag"].astype(bool)

    tp = int(((predicted) & (ground_truth)).sum())
    fp = int(((predicted) & (~ground_truth)).sum())
    fn = int(((~predicted) & (ground_truth)).sum())
    tn = int(((~predicted) & (~ground_truth)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(sessions)

    print("=== Prototype vs Baseline: Measurable Experiment ===\n")
    print(f"Total sessions evaluated: {len(sessions)}")
    print(f"True mismatches (ground truth): {int(ground_truth.sum())}\n")

    print("--- Baseline (attendance-only) ---")
    print("Mismatches it can detect: 0 (concept doesn't exist in this model)")
    print("Precision / Recall / F1: N/A -- not applicable, no prediction made\n")

    print("--- Prototype (semantic + rating-divergence detector) ---")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True negatives:  {tn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 score:  {f1:.2f}")
    print(f"Accuracy:  {accuracy:.2f}")

    sessions.to_csv("experiment_results_detailed.csv", index=False)
    with open("experiment_results_summary.md", "w") as f:
        f.write("# Experiment results: Prototype vs Baseline\n\n")
        f.write(f"- Total sessions: {len(sessions)}\n")
        f.write(f"- True mismatches (ground truth): {int(ground_truth.sum())}\n\n")
        f.write("## Baseline (attendance-only)\n")
        f.write("- Mismatches detected: 0 (no such concept exists in the model)\n\n")
        f.write("## Prototype (semantic + rating-divergence detector)\n")
        f.write(f"- Precision: {precision:.2f}\n- Recall: {recall:.2f}\n- F1: {f1:.2f}\n- Accuracy: {accuracy:.2f}\n")
        f.write(f"- TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")

    print("\nResults written to experiment_results_detailed.csv and experiment_results_summary.md")


if __name__ == "__main__":
    main()
