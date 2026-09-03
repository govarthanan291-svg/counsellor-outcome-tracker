# Collaborative Outcome Tracker -- Prototype Setup

This prototype uses a **local SLM (via Ollama)** and **sentence-transformers**
for semantic mismatch detection between what a client says and what actually
happened in their sessions. Both need to be installed on your own machine
with internet access -- they will NOT work inside a restricted/offline
environment, since the models are downloaded from the internet the first
time they run.

## 1. Install Python dependencies

From the `prototype/` folder:

```bash
pip install streamlit pandas altair sentence-transformers requests
```

The first time you run the app, `sentence-transformers` will download the
`all-MiniLM-L6-v2` model (~90MB) from Hugging Face automatically. This needs
internet access once -- after that it's cached locally and works offline.

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

**Note:** If Ollama isn't running, the app still works -- the mismatch
*flagging* (the actual decision logic) doesn't depend on Ollama at all. Only
the plain-language explanation shown to the counsellor will show an
"[Explanation unavailable]" message instead. This is intentional -- see the
failure-mode notes in `mismatch_detector.py`.

## 3. Folder layout expected

Make sure the folder structure looks like this (the app reads synthetic data
via a relative path):

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

This runs two example sessions (one clean, one deliberately mismatched) and
prints the detector's output as JSON.

## Why this design (for your documentation)

- The **flagging decision** is a simple, inspectable rule based on semantic
  similarity + rating/text divergence -- not a black-box LLM judgment. This
  matters for the failure-mode analysis: you can explain exactly why any
  session was or wasn't flagged.
- The **SLM (Ollama)** is used only to turn an already-made decision into a
  readable sentence for the counsellor -- it never decides the outcome
  itself, and the system degrades gracefully if it's unavailable.
- All flags route to a **human counsellor** for review (Stage 4 in the
  field-workflow map) -- the AI never concludes anything about a client's
  actual wellbeing on its own.
