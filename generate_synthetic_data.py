"""
Synthetic data generator for the Collaborative Outcome Tracker project.

Generates three linked datasets:
  - clients.csv     : client demographics + urgency level
  - goals.csv        : client-defined goals (baseline, target, target date)
  - sessions.csv      : session-by-session self-reported progress, barriers,
                        and agreed adjustments (with deliberately injected
                        mismatches for testing the AI mismatch-detection layer)

Run: python3 generate_synthetic_data.py
Output: CSV files written to ./synthetic_data/
"""

import csv
import os
import random
from datetime import date, timedelta

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

OUT_DIR = "synthetic_data"
os.makedirs(OUT_DIR, exist_ok=True)

N_CLIENTS = 40

URGENCY_LEVELS = ["low", "moderate", "high"]
URGENCY_WEIGHTS = [0.5, 0.35, 0.15]  # most clients are low/moderate urgency

PRESENTING_CONCERNS = [
    "exam anxiety", "academic stress", "social isolation", "low mood",
    "sleep difficulties", "relationship conflict", "burnout", "panic attacks",
    "homesickness", "time management struggles", "grief", "self-esteem issues",
]

GOAL_TEMPLATES = [
    ("Reduce exam-related anxiety", "Panic before every exam (8/10 anxiety)", "Manageable anxiety (3/10) before exams"),
    ("Improve sleep consistency", "Sleeping 4-5 hours, irregular schedule", "Sleeping 7 hours on a consistent schedule"),
    ("Build one supportive friendship", "No close friends on campus", "At least one person to talk to weekly"),
    ("Reduce daily low mood", "Low mood most days, low motivation", "Low mood fewer than 2 days a week"),
    ("Manage panic attacks", "2-3 panic attacks per week", "Panic attacks reduced to under 1 per month"),
    ("Improve study-life balance", "Studying till 2am most nights, no breaks", "Fixed study hours with evening downtime"),
    ("Process grief around a loss", "Overwhelmed by grief, avoiding routine", "Able to engage in daily routine most days"),
    ("Increase self-esteem", "Frequent self-critical thoughts", "Fewer self-critical thoughts, more self-compassion"),
]

BARRIER_OPTIONS = [
    "missed sessions due to coursework deadlines",
    "family obligations limited practice time",
    "relapse into old coping habits under stress",
    "difficulty applying strategies during actual exam week",
    "roommate conflict added new stress",
    "financial stress distracted from goal work",
    "part-time job hours increased, less time to rest",
    "reluctance to open up about the real issue",
    "no barriers this session",
    "physical illness disrupted routine",
]

ADJUSTMENT_OPTIONS = [
    "broke the goal into smaller weekly steps",
    "added a grounding technique for acute moments",
    "shifted target date by two weeks",
    "introduced a daily check-in journal",
    "reduced session focus to one barrier at a time",
    "no adjustment needed, continuing as planned",
    "involved a peer support group as additional support",
    "revised goal to be more realistic given workload",
]

PROGRESS_PHRASES_POSITIVE = [
    "feeling much better this week",
    "made real progress on this",
    "things are improving steadily",
    "had a genuinely good week",
]
PROGRESS_PHRASES_NEUTRAL = [
    "about the same as last week",
    "some ups and downs",
    "mixed week overall",
]
PROGRESS_PHRASES_NEGATIVE = [
    "struggled a lot this week",
    "felt worse than before",
    "barely managed to cope",
]


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 0)))


def generate_clients(n):
    clients = []
    for i in range(1, n + 1):
        clients.append({
            "client_id": f"C{i:03d}",
            "client_name": fake.first_name() + " " + fake.last_name()[0] + ".",  # anonymized but human
            "age": random.randint(18, 26),
            "urgency_level": random.choices(URGENCY_LEVELS, weights=URGENCY_WEIGHTS)[0],
            "presenting_concern": random.choice(PRESENTING_CONCERNS),
            "intake_date": random_date(date(2026, 1, 6), date(2026, 3, 1)).isoformat(),
        })
    return clients


def generate_goals(clients):
    goals = []
    goal_id = 1
    for client in clients:
        n_goals = random.choices([1, 2], weights=[0.7, 0.3])[0]
        chosen = random.sample(GOAL_TEMPLATES, n_goals)
        for goal_text, baseline, target in chosen:
            created = date.fromisoformat(client["intake_date"]) + timedelta(days=random.randint(1, 5))
            goals.append({
                "goal_id": f"G{goal_id:04d}",
                "client_id": client["client_id"],
                "goal_text": goal_text,
                "baseline": baseline,
                "target": target,
                "created_date": created.isoformat(),
                "target_date": (created + timedelta(weeks=random.choice([6, 8, 10, 12]))).isoformat(),
            })
            goal_id += 1
    return goals


def generate_sessions(goals):
    sessions = []
    session_id = 1
    for goal in goals:
        n_sessions = random.randint(3, 8)
        session_date = date.fromisoformat(goal["created_date"])
        for s in range(n_sessions):
            session_date = session_date + timedelta(days=random.randint(6, 10))

            # progress trajectory: mostly trends toward improvement but not always
            rating = random.choices(
                [2, 3, 4, 5, 6, 7, 8, 9],
                weights=[5, 8, 10, 15, 18, 18, 15, 11],
            )[0]

            if rating >= 7:
                progress_text = random.choice(PROGRESS_PHRASES_POSITIVE)
            elif rating >= 4:
                progress_text = random.choice(PROGRESS_PHRASES_NEUTRAL)
            else:
                progress_text = random.choice(PROGRESS_PHRASES_NEGATIVE)

            barrier = random.choice(BARRIER_OPTIONS)
            adjustment = random.choice(ADJUSTMENT_OPTIONS)

            # --- Deliberate mismatch injection (~15% of sessions) ---
            # Client's numeric rating and text/barrier description contradict
            # each other -- this is what the Stage 4 AI layer should flag.
            is_mismatch = random.random() < 0.15
            if is_mismatch:
                if rating >= 7:
                    # High rating but barrier text describes serious struggle
                    barrier = random.choice([
                        "relapse into old coping habits under stress",
                        "reluctance to open up about the real issue",
                        "physical illness disrupted routine",
                    ])
                    progress_text = "feeling much better this week, though honestly still struggling most days"
                else:
                    # Low rating but text sounds falsely upbeat
                    progress_text = "feeling much better this week"
                    barrier = "no barriers this session"

            sessions.append({
                "session_id": f"S{session_id:05d}",
                "goal_id": goal["goal_id"],
                "client_id": goal["client_id"],
                "session_date": session_date.isoformat(),
                "self_reported_rating": rating,
                "self_reported_text": progress_text,
                "barriers": barrier,
                "agreed_adjustment": adjustment,
                "flagged_mismatch_ground_truth": is_mismatch,  # for evaluating the AI later, not shown to counsellor
            })
            session_id += 1
    return sessions


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_goal_status(goal_sessions):
    """Derive a goal's current status from its own session history --
    not an arbitrary label, so it stays consistent with what's actually
    logged."""
    if not goal_sessions:
        return "pending"
    recent = goal_sessions[-3:]
    avg_recent = sum(s["self_reported_rating"] for s in recent) / len(recent)
    if avg_recent >= 8:
        return "achieved"
    elif avg_recent <= 4:
        return "needs revision"
    return "active"


def main():
    clients = generate_clients(N_CLIENTS)
    goals = generate_goals(clients)
    sessions = generate_sessions(goals)

    sessions_by_goal = {}
    for s in sessions:
        sessions_by_goal.setdefault(s["goal_id"], []).append(s)
    for goal in goals:
        goal["status"] = compute_goal_status(sessions_by_goal.get(goal["goal_id"], []))

    write_csv(os.path.join(OUT_DIR, "clients.csv"), clients,
              ["client_id", "client_name", "age", "urgency_level", "presenting_concern", "intake_date"])
    write_csv(os.path.join(OUT_DIR, "goals.csv"), goals,
              ["goal_id", "client_id", "goal_text", "baseline", "target", "created_date", "target_date", "status"])
    write_csv(os.path.join(OUT_DIR, "sessions.csv"), sessions,
              ["session_id", "goal_id", "client_id", "session_date", "self_reported_rating",
               "self_reported_text", "barriers", "agreed_adjustment", "flagged_mismatch_ground_truth"])

    print(f"Generated {len(clients)} clients, {len(goals)} goals, {len(sessions)} sessions")
    n_mismatch = sum(1 for s in sessions if s["flagged_mismatch_ground_truth"])
    print(f"Deliberately injected mismatches: {n_mismatch} ({n_mismatch/len(sessions):.1%} of sessions)")
    print(f"Files written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
