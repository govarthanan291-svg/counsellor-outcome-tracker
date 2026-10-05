# Collaborative Outcome Tracker -- Prototype Setup

This prototype flags mismatches with a **rating-drop check** and a
**keyword check**. Those rules run with no extra models.

A **sentence-transformers** semantic check exists in `mismatch_detector.py`
but is **off by default** (`USE_SEMANTIC_CHECK = False`) after a real-model
evaluation found it hurt precision. A **local SLM via Ollama** is optional
and only writes counsellor-facing explanations of flags that were already
made. Neither is required on Streamlit Cloud.

## 1. Install Python dependencies

From the `prototype/` folder:

```bash
pip install streamlit pandas altair sentence-transformers requests
```

`sentence-transformers` is only downloaded if you set
`USE_SEMANTIC_CHECK = True`. The default app path does not load that model.

## 2. Install and set up Ollama (for AI explanations)

1. Download and install Ollama: https://ollama.com/download
2. Pull a small model (either works, the code defaults to the first one):
   ```bash
   ollama pull llama3.2:1b
   # or, if you prefer:
   ollama pull phi3:mini
   ```
   If you use `phi3:mini`, change `OLLAMA_MODEL` at the top of
   `mismatch_detector.py` to `"phi3:mini"`.
3. Ollama usually starts its local server automatically after install. If not:
   ```bash
   ollama serve
   ```
   It should be reachable at `http://localhost:11434`.

**Note:** If Ollama isn't running (including Streamlit Cloud), leave the
"Generate Ollama explanations" checkbox off. If you turn it on anyway, the
detector tries once (5s timeout), then skips further calls for that process
and shows a short "SLM not reachable" note. Flagging still works. Do not
rely on `localhost:11434` in Cloud.

## 3. Folder layout expected

Make sure the folder structure looks like this. `app.py` resolves
`synthetic_data/` from its own file path (not the process working
directory), so Streamlit Cloud is fine as long as the full repo is deployed:

```
counsellor-outcome-tracker/
├── synthetic_data/
│   ├── clients.csv
│   ├── goals.csv
│   └── sessions.csv
└── prototype/
    ├── app.py
    ├── mismatch_detector.py
    └── README.md   (this file)
```

## 4. Run it

From inside `prototype/`:

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Pick a
client from the dropdown, expand a goal, and click "Run mismatch check" to
see the AI flag sessions where self-reported progress and barriers don't
line up -- with a plain-language explanation for the counsellor.

## 5. Quick standalone test (without Streamlit)

To sanity-check the detector logic on its own, from `prototype/`:

```bash
python3 mismatch_detector.py
```

This runs three example sessions (clean, keyword mismatch, rating-drop)
and prints the detector's output as JSON.

## Tests

From the **repo root** (not `prototype/`):

```bash
pytest tests/ -v
```

## Why this design (for your documentation)

- The **flagging decision** is a simple, inspectable rule (rating-drop +
  keyword, optional semantic check) -- not a black-box LLM judgment.
- The **SLM (Ollama)** only turns an already-made decision into a readable
  sentence. It never decides the outcome, and it is skipped after one
  failed contact if the server is down.
- All flags route to a **human counsellor** for review (Stage 4 in the
  field-workflow map) -- the AI never concludes anything about a client's
  actual wellbeing on its own.
