"""
app.py -- Collaborative Outcome Tracker (Streamlit prototype)
Professional, calm visual design suited to a counselling context.

Run locally:
    streamlit run app.py

Requirements:
    pip install streamlit pandas altair sentence-transformers requests

Expects synthetic_data/{clients,goals,sessions}.csv as a sibling of this
folder (repo-root/synthetic_data/), resolved from this file's location so
the working directory on Streamlit Cloud does not matter.
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

from mismatch_detector import MismatchDetector, USE_SEMANTIC_CHECK

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Outcome Tracker", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#12181F"
SURFACE = "#1A222C"
SAGE = "#6FA98D"
SAGE_DIM = "#4C7D66"
AMBER = "#E8A857"
TEXT = "#E7EAEE"
TEXT_MUTED = "#8C97A6"
BORDER = "#2C3744"
CORAL = "#D9765A"

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
.stApp {{
    background-color: {INK};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Lora', serif;
    font-weight: 600;
    color: {TEXT};
}}
.hero-title {{
    font-family: 'Lora', serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 0.1rem;
}}
.hero-sub {{
    font-family: 'Inter', sans-serif;
    color: {TEXT_MUTED};
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}}
.client-line {{
    font-family: 'Inter', sans-serif;
    color: {TEXT_MUTED};
    font-size: 0.9rem;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 0.9rem;
    margin-bottom: 0.4rem;
}}
.urgency-low {{ color: {SAGE}; font-weight: 600; }}
.urgency-moderate {{ color: {AMBER}; font-weight: 600; }}
.urgency-high {{ color: {CORAL}; font-weight: 600; }}

.goal-block {{
    border-left: 3px solid {SAGE_DIM};
    padding: 0.3rem 0 0 1.1rem;
    margin: 1.6rem 0 1.1rem 0;
}}
.goal-title {{
    font-family: 'Lora', serif;
    font-size: 1.25rem;
    color: {TEXT};
    margin-bottom: 0.3rem;
}}
.goal-meta {{
    font-size: 0.87rem;
    color: {TEXT_MUTED};
    line-height: 1.6;
}}
.goal-meta b {{ color: {TEXT}; font-weight: 500; }}

.flag-row {{
    border-left: 3px solid {AMBER};
    background-color: {SURFACE};
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    border-radius: 0 4px 4px 0;
}}
.flag-row.sev-high {{ border-left-color: {CORAL}; }}
.flag-row.sev-medium {{ border-left-color: {AMBER}; }}
.flag-row.sev-low {{ border-left-color: {SAGE_DIM}; }}
.flag-header {{
    font-size: 0.88rem;
    color: {AMBER};
    font-weight: 600;
    margin-bottom: 0.35rem;
}}
.flag-detail {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    line-height: 1.7;
}}
.flag-detail b {{ color: {TEXT}; font-weight: 500; }}
.review-note {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    font-style: italic;
    margin-top: 0.4rem;
}}

.summary-line {{
    font-size: 0.95rem;
    color: {TEXT};
    margin: 0.8rem 0;
    padding: 0.7rem 1rem;
    background-color: {SURFACE};
    border-radius: 4px;
}}

div[data-testid="stSelectbox"] label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.85rem;
}}
.stButton > button {{
    background-color: {SAGE_DIM};
    color: {TEXT};
    border: none;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
}}
.stButton > button:hover {{
    background-color: {SAGE};
    color: {INK};
}}
[data-testid="stExpander"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
.status-badge {{
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 10px;
    margin-left: 0.6rem;
}}
.status-active {{ background-color: rgba(111,169,141,0.18); color: {SAGE}; }}
.status-achieved {{ background-color: rgba(111,169,141,0.32); color: {SAGE}; }}
.status-needs-revision {{ background-color: rgba(232,168,87,0.22); color: {AMBER}; }}
.status-pending {{ background-color: rgba(140,151,166,0.2); color: {TEXT_MUTED}; }}
.status-high {{ background-color: rgba(217,118,90,0.22); color: {CORAL}; }}
.status-medium {{ background-color: rgba(232,168,87,0.22); color: {AMBER}; }}
.status-low {{ background-color: rgba(111,169,141,0.18); color: {SAGE}; }}

.ai-tag {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 0.3rem 0.7rem;
    display: inline-block;
    margin-top: 0.3rem;
}}
.ai-tag b {{ color: {SAGE}; font-weight: 600; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


def resolve_data_dir() -> str:
    """Find synthetic_data from this file's location, not the process cwd.

    Streamlit Cloud typically runs with the repo root as cwd, so a bare
    `../synthetic_data` path would miss. Prefer the sibling of prototype/,
    then a copy bundled next to app.py.
    """
    candidates = [
        os.path.normpath(os.path.join(SCRIPT_DIR, "..", "synthetic_data")),
        os.path.join(SCRIPT_DIR, "synthetic_data"),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "sessions.csv")):
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(
        "Could not find synthetic_data/sessions.csv. Looked in: " + searched
    )


DATA_DIR = resolve_data_dir()


@st.cache_data
def load_data():
    clients = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
    goals = pd.read_csv(os.path.join(DATA_DIR, "goals.csv"))
    sessions = pd.read_csv(os.path.join(DATA_DIR, "sessions.csv"))
    return clients, goals, sessions


@st.cache_resource
def load_detector(use_ollama: bool):
    return MismatchDetector(use_ollama=use_ollama)


def analyse_sessions(detector, goal_sessions):
    ratings = goal_sessions["self_reported_rating"].tolist()
    results = []
    for i, (_, row) in enumerate(goal_sessions.iterrows()):
        results.append(detector.check_session(row.to_dict(), rating_history=ratings[:i]))
    return results


def severity_badge_class(severity: str) -> str:
    return {"high": "status-high", "medium": "status-medium", "low": "status-low"}.get(
        severity, "status-pending"
    )


clients, goals, sessions = load_data()

st.markdown('<div class="hero-title">Outcome Tracker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Progress measured against goals the client defined — not attendance.</div>',
    unsafe_allow_html=True,
)

semantic_note = (
    "optional sentence-transformers check is <b>off</b> by default "
    "(measured to hurt precision — see experiments/experiment_results.md)"
    if not USE_SEMANTIC_CHECK
    else "sentence-transformers semantic check is <b>on</b>"
)
st.markdown(
    f'<div class="ai-tag">Mismatch flags use a rating-drop check and a keyword check; '
    f'{semantic_note}. Ollama explanations are optional and skipped if the local SLM '
    f'is not running. No client data leaves this machine.</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

client_ids = clients["client_id"].tolist()
client_labels = {row["client_id"]: f'{row["client_id"]} — {row["client_name"]}' for _, row in clients.iterrows()}
selected_client = st.selectbox(
    "Client", client_ids, format_func=lambda cid: client_labels[cid], label_visibility="collapsed"
)

client_row = clients[clients["client_id"] == selected_client].iloc[0]
client_goals = goals[goals["client_id"] == selected_client]

urgency = client_row["urgency_level"]
urgency_class = f"urgency-{urgency}"
st.markdown(
    f'<div class="client-line"><b style="color:{TEXT}; font-size:1.05rem;">{client_row["client_name"]}</b> '
    f'<span style="color:{TEXT_MUTED};">({selected_client})</span> '
    f'&nbsp;·&nbsp; <span class="{urgency_class}">{urgency} urgency</span> '
    f'&nbsp;·&nbsp; {client_row["presenting_concern"]}</div>',
    unsafe_allow_html=True,
)

use_ollama = st.checkbox(
    "Generate Ollama explanations (needs a local `ollama serve`; ignored on Streamlit Cloud)",
    value=False,
)
severity_filter = st.selectbox(
    "Show flags",
    ["all", "high", "medium", "low"],
    format_func=lambda v: {
        "all": "All severities",
        "high": "High only",
        "medium": "Medium only",
        "low": "Low only",
    }[v],
)

detector = load_detector(use_ollama)


def results_for_goal(goal_id, goal_sessions):
    """Keep per-goal results in session state so toggling the filter does
    not re-call Ollama (5s timeout × N flags) on every rerun."""
    key = f"mismatch::{selected_client}::{goal_id}::{use_ollama}"
    if key not in st.session_state:
        st.session_state[key] = analyse_sessions(detector, goal_sessions)
    return st.session_state[key]


if client_goals.empty:
    st.markdown('<div class="summary-line">No goals recorded for this client yet.</div>', unsafe_allow_html=True)
else:
    per_goal_results = {}
    all_results = []
    for _, goal in client_goals.iterrows():
        goal_sessions = sessions[sessions["goal_id"] == goal["goal_id"]].sort_values("session_date")
        if goal_sessions.empty:
            per_goal_results[goal["goal_id"]] = (goal_sessions, [])
            continue
        results = results_for_goal(goal["goal_id"], goal_sessions)
        per_goal_results[goal["goal_id"]] = (goal_sessions, results)
        all_results.extend(results)

    flagged = [r for r in all_results if r.is_flagged]
    high_n = sum(1 for r in flagged if r.severity == "high")
    med_n = sum(1 for r in flagged if r.severity == "medium")
    low_n = sum(1 for r in flagged if r.severity == "low")
    st.markdown(
        f'<div class="summary-line"><b>{len(flagged)}</b> of <b>{len(all_results)}</b> sessions '
        f'flagged for review &nbsp;·&nbsp; {high_n} high &nbsp;·&nbsp; {med_n} medium &nbsp;·&nbsp; {low_n} low'
        f'</div>',
        unsafe_allow_html=True,
    )

    for _, goal in client_goals.iterrows():
        status = goal.get("status", "active")
        status_class = f"status-{status.replace(' ', '-')}"
        st.markdown(f"""
        <div class="goal-block">
            <div class="goal-title">{goal['goal_text']} <span class="status-badge {status_class}">{status}</span></div>
            <div class="goal-meta">
                <b>Baseline</b> — {goal['baseline']}<br>
                <b>Target</b> — {goal['target']} &nbsp;·&nbsp; by {goal['target_date']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        goal_sessions, results = per_goal_results[goal["goal_id"]]

        if goal_sessions.empty:
            st.markdown('<div class="summary-line">No sessions logged yet.</div>', unsafe_allow_html=True)
            continue

        chart = (
            alt.Chart(goal_sessions)
            .mark_line(point=alt.OverlayMarkDef(color=SAGE, size=45), color=SAGE, strokeWidth=2.2)
            .encode(
                x=alt.X("session_date:T", title=None, axis=alt.Axis(labelColor=TEXT_MUTED, gridColor=BORDER)),
                y=alt.Y("self_reported_rating:Q", title="self-reported rating",
                        axis=alt.Axis(labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, gridColor=BORDER),
                        scale=alt.Scale(domain=[0, 10])),
            )
            .properties(height=200, background=INK)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

        flagged_here = [r for r in results if r.is_flagged]
        visible = [
            r for r in flagged_here
            if severity_filter == "all" or r.severity == severity_filter
        ]
        st.caption(f"{len(flagged_here)} flagged in this goal" + (
            f" ({len(visible)} shown)" if severity_filter != "all" else ""
        ))

        for r in sorted(visible, key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r.severity, 3)):
            session_row = goal_sessions[goal_sessions["session_id"] == r.session_id].iloc[0]
            explanation = r.explanation or "No generated explanation (Ollama not requested)."
            sev_class = severity_badge_class(r.severity)
            st.markdown(f"""
            <div class="flag-row sev-{r.severity}">
                <div class="flag-header">{r.session_id} — {session_row['session_date']} &nbsp;
                    <span class="status-badge {sev_class}">{r.severity} priority</span>
                </div>
                <div class="flag-detail">
                    <b>Rating</b> {session_row['self_reported_rating']}/10 &nbsp;·&nbsp;
                    <b>Reported</b> "{session_row['self_reported_text']}"<br>
                    <b>Barriers</b> {session_row['barriers']}<br>
                    <b>Why flagged</b> {', '.join(r.reasons)}<br>
                    <b>Counsellor note</b> {explanation}
                </div>
                <div class="review-note">Flagged for human review only — no conclusion is drawn automatically.</div>
            </div>
            """, unsafe_allow_html=True)

        log = goal_sessions[["session_id", "session_date", "self_reported_rating",
                             "self_reported_text", "barriers", "agreed_adjustment"]].copy()
        by_id = {r.session_id: r for r in results}
        log["severity"] = log["session_id"].map(lambda sid: by_id[sid].severity if sid in by_id else "none")
        if severity_filter != "all":
            log = log[log["severity"] == severity_filter]

        with st.expander("Full session log"):
            st.dataframe(log, use_container_width=True, hide_index=True)
