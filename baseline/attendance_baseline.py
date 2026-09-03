"""
Baseline: Attendance-only outcome tracker.

This mimics the CURRENT real-world approach at the counselling centre --
tracking only session counts / attendance, with no notion of client-defined
goals, progress, barriers, or adjustments. It exists purely as a comparison
point for the main prototype (Step 4), to demonstrate what information is
lost when outcome tracking is reduced to attendance alone.

Run: python3 attendance_baseline.py
Reads from: ../synthetic_data/{clients,goals,sessions}.csv
Writes to: ./baseline_report.csv
"""

import csv
import os
from collections import defaultdict

DATA_DIR = os.path.join("..", "synthetic_data")


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    clients = load_csv("clients.csv")
    sessions = load_csv("sessions.csv")

    # This is the entire "model": count sessions attended per client.
    session_counts = defaultdict(int)
    for s in sessions:
        session_counts[s["client_id"]] += 1

    report = []
    for c in clients:
        count = session_counts.get(c["client_id"], 0)
        # The only "outcome" the baseline can produce: engagement level,
        # inferred crudely from session count. No goal, no progress, no
        # evidence of actual improvement.
        if count >= 6:
            engagement = "high engagement"
        elif count >= 3:
            engagement = "moderate engagement"
        else:
            engagement = "low engagement"

        report.append({
            "client_id": c["client_id"],
            "client_name": c["client_name"],
            "sessions_attended": count,
            "engagement_level": engagement,
            "outcome_conclusion": "N/A - attendance only, no goal-based outcome available",
        })

    out_path = "baseline_report.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "client_id", "client_name", "sessions_attended", "engagement_level", "outcome_conclusion"
        ])
        writer.writeheader()
        writer.writerows(report)

    print(f"Baseline report written to {out_path}")
    print(f"Clients processed: {len(report)}")
    print()
    print("--- Sample output ---")
    for row in report[:5]:
        print(row)
    print()
    print("LIMITATION: This baseline cannot answer whether any client actually")
    print("improved. It can only say how often they showed up. Two clients with")
    print("identical attendance could have completely opposite real outcomes --")
    print("this baseline has no way to tell them apart.")


if __name__ == "__main__":
    main()
