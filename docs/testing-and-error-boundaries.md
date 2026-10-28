# Testing & Error Boundaries

This document describes (1) what the automated tests cover and how they are
designed, and (2) every point where the system can receive bad input or lose a
dependency — what happens there, why, and which test proves it. It also lists
what is **not** tested, so the coverage claim is honest.

Scope: the decision layer in `prototype/mismatch_detector.py`, which is the
only component with automated tests. Suite size: **54 tests, ~0.2 s, no
network, no model download.**

## 1. Running the tests

```bash
pip install pytest requests        # requests is imported by the detector
pytest tests/ -v                   # everything
pytest tests/ -k Ollama            # one area
pytest tests/ -k "Boundaries"      # input + Ollama boundary tests
```

Expected: `54 passed`. Nothing needs Ollama running, and the embedding model
is never loaded (the semantic check is off by default and loading is lazy —
`test_model_is_not_loaded_on_init_or_check` asserts `_load_model` is not called
and `_model` stays `None`).

## 2. Test design principles

| Principle | How it shows up |
|---|---|
| **Offline & deterministic** | The one network call (`requests.post` to Ollama) is patched in every test that could reach it. No test depends on a model, a port, or the clock. |
| **Boundary-value testing** | Every threshold has a test *at* the edge and one *just inside*: rating 7 vs 6 (keyword rule), drop of exactly 3 vs just under (drop rule), drop of exactly 5 vs just under (high severity). |
| **Decision/explanation separation is tested** | Ollama failures are asserted *not* to change `is_flagged` or `severity`, because flagging is decided before the explanation step. |
| **Regression tests tied to real findings** | The blank-barrier tests exist because a real crash was found (section 4). They were run against the pre-fix code and fail there (see 2.1). |
| **Fail-loud vs tolerate is explicit** | Each bad-input case is deliberately classified as *tolerated* or *raises*, and a test pins the choice. A silently-wrong clinical flag is judged worse than a visible error. |
| **Known gaps are labelled** | A test that documents unwanted-but-current behaviour is marked as a *characterization* test in its comment, so it is not mistaken for a requirement. |

### 2.1 Regression check on the NaN fix

The new `TestInputBoundaries` / `TestOllamaBoundaries` tests were run against
the **pre-fix** `mismatch_detector.py`. Result: 4 failed, 50 passed.

| Failing test (pre-fix code) | What it proves |
|---|---|
| `test_blank_barriers_do_not_crash_at_high_rating[nan]` | Real bug: `AttributeError` on a blank barrier cell with rating ≥ 7 |
| `test_blank_barriers_do_not_crash_at_high_rating[None]` | Same crash for `None` |
| `test_blank_barriers_never_reach_the_prompt_as_the_word_nan` | Real bug: a blank cell was sent to the SLM as the literal word "nan" |
| `test_missing_required_key_raises_key_error[self_reported_text]` | **Intentional behaviour change**, not a bug fix: a missing `self_reported_text` now always raises; before, it raised only when a flagged session reached the Ollama prompt (data-dependent, hence inconsistent) |

## 3. Test inventory

| Class | Tests | What it covers |
|---|---:|---|
| `TestSemanticMismatch` | 3 | Optional similarity rule: no-barrier shortcut, low similarity flags, high similarity does not (model mocked) |
| `TestRatingTextMismatch` | 5 | Keyword rule: keyword + high rating flags; neutral barrier / low rating never; rating exactly 7 flags; 6 does not |
| `TestRatingDropMismatch` | 8 | Drop rule: no history, sharp drop, exactly-at and just-below threshold, small fluctuation, only last 3 ratings used, rises never flag, single prior rating |
| `TestSeverityTiers` | 6 | none / low / medium / high mapping incl. the 5-point boundary and two-signal escalation |
| `TestSemanticCheckDisabled` | 3 | With the flag off the semantic method is not called and the model is never loaded; forced on, it is called |
| `TestCheckSessionIntegration` | 5 | End-to-end `check_session`: clean session, large drop alone is high, first session works, Ollama off = no network call, Ollama failure is cached and not retried |
| `TestInputBoundaries` | 15 | Blank/NaN/None/whitespace cells, numeric-string rating, case-insensitive keywords, `None` vs `[]` history, non-numeric rating raises, each missing key raises, out-of-range rating (known gap) |
| `TestOllamaBoundaries` | 9 | Connection error, timeout, HTTP error, malformed JSON, 404 from a live server, empty response, success path, no "nan" in prompt, unflagged sessions never call Ollama |
| **Total** | **54** | |

(Counts are pytest items, so parametrized cases count individually.)

## 4. Error-boundary table

Reading guide — **Tolerated**: handled, the app keeps working. **Raises**: fails
visibly on purpose. **Gap**: not handled today.

| # | Boundary | Trigger | Behaviour | Why | Test |
|---|---|---|---|---|---|
| 1 | Blank barrier/text cell | Empty CSV cell → pandas `NaN`; or `None`, `""`, whitespace | **Tolerated** — treated as "no text" | Counsellors leave fields blank; was a crash for `NaN`/`None` at rating ≥ 7 | `test_blank_barriers_do_not_crash_at_high_rating` (×4), `test_blank_self_reported_text_is_tolerated` |
| 2 | Blank cell → SLM prompt | Flagged session with blank barrier, Ollama on | **Tolerated** — prompt shows `""`, not `nan` | Garbage in the prompt degrades the explanation | `test_blank_barriers_never_reach_the_prompt_as_the_word_nan` |
| 3 | Masked distress with no barrier text | Rating drops, text positive, barrier blank | **Tolerated and flagged** by the drop rule | This is failure mode 1 (client C027); the drop rule needs no barrier text | `test_blank_barriers_still_allow_rating_drop_to_flag` |
| 4 | Numeric string rating | `"8"` | **Tolerated** (`int()` coercion) | CSV/form values often arrive as strings | `test_numeric_string_rating_is_accepted` |
| 5 | Non-numeric rating | `"eight"` | **Raises** `ValueError` | A guessed rating would produce a silently wrong flag | `test_non_numeric_rating_raises_value_error` |
| 6 | Missing required key | `session_id`, `self_reported_rating`, `self_reported_text`, or `barriers` absent | **Raises** `KeyError` | Schema error should surface, not be skipped | `test_missing_required_key_raises_key_error` (×4) |
| 7 | No rating history | First session, `None`, or `[]` | **Tolerated** — drop rule skipped | A first session cannot "drop" | `test_no_history_does_not_flag`, `test_none_history_behaves_like_empty_history`, `test_missing_rating_history_still_works` |
| 8 | Short history | 1–2 prior ratings | **Tolerated** — average over what exists | Early sessions should still be checkable | `test_single_prior_rating_still_computes_average` |
| 9 | Long history | > 3 prior ratings | **Tolerated** — only last 3 used | Compare to *recent* baseline, not all-time | `test_uses_only_last_three_ratings` |
| 10 | Threshold edges | Rating 7/6; drop 3/just under; drop 5/just under | Exact-edge value flags, just-under does not | Pins the `>=` semantics | `TestRatingTextMismatch`, `TestRatingDropMismatch`, `TestSeverityTiers` |
| 11 | Ollama unreachable | Not installed / not running (normal on Streamlit Cloud) | **Tolerated** — fallback note; flag and severity unchanged | Explanation is a nice-to-have | `test_any_failure_falls_back_and_keeps_the_flag[ConnectionError]` |
| 12 | Ollama hangs | Timeout (5 s) | **Tolerated** — same fallback | A hung local server must not stall a session | `…[Timeout]` |
| 13 | Ollama bad response | HTTP error status, malformed JSON, 404 model-not-found | **Tolerated** — same fallback | Any failure must not break flagging | `…[HTTPError]`, `…[ValueError]`, `test_http_error_status_from_a_live_server_also_falls_back` |
| 14 | Ollama blank reply | Model returns whitespace | **Tolerated** — `explanation = None` | Avoid showing an empty box | `test_empty_model_response_yields_none_not_blank_string` |
| 15 | Ollama failure repeated | Many flags after the first failure | **Tolerated** — one failed call, then no retries (latch) | 30 flags must cost one timeout, not 30 | `test_ollama_failure_is_cached_and_not_retried` |
| 16 | Explanations disabled / not needed | `use_ollama=False`, or session not flagged | **No network call at all** | Privacy + speed | `test_ollama_off_does_not_call_network`, `test_unflagged_session_never_calls_ollama` |
| 17 | Semantic model | Check off (default) | **Model never loaded** | Cloud start-up, offline tests | `TestSemanticCheckDisabled` |
| 18 | Data folder missing | `synthetic_data/sessions.csv` not found | **Raises** `FileNotFoundError` listing the paths searched (`app.py: resolve_data_dir`) | A clear message beats a pandas traceback | *not unit-tested* |
| 19 | Out-of-range rating | e.g. `15` | **Gap** — accepted as a "high" rating | No range validation | `test_out_of_range_rating_is_currently_not_validated` (characterization) |

How the layers fit: bad data is caught or normalised **first** (`_as_text`,
`int()`), the three rules decide **next**, severity is derived from the rules,
and only then is the optional explanation attempted — in its own `try` block,
so nothing in the explanation step can change or lose a flag.

## 5. What is NOT tested (honest scope)

- **The Streamlit UI (`prototype/app.py`)** has no committed automated tests. It
  was exercised manually and in one ad-hoc Streamlit `AppTest` run that is not
  part of the repo. A UI regression would not be caught by `pytest`.
- **`generate_synthetic_data.py`, `baseline/`, `experiments/`** have no unit
  tests. They are reproducible instead: the generator is seeded (42), and
  `experiments/experiment.py` re-derives the published precision/recall/F1.
- **The real embedding model.** Unit tests mock it. Its behaviour was measured
  in `experiments/` (that measurement is why the semantic check is off), not
  asserted in tests.
- **Data-file schema.** Missing or renamed CSV columns surface as pandas
  `KeyError`s; there is no explicit schema validation (see README, *Data schema*).
- **Rating range.** Row 19 above.
- **Non-English barrier text.** The keyword rule is an English-only substring
  match, so a barrier written only in Tamil/Tanglish will not trigger it. The
  rating-drop rule is language-independent and still applies. This is the
  practical face of failure mode 2.
- **Explanation quality.** Tests prove a failing SLM cannot break flagging;
  they cannot prove the SLM's sentence is accurate (failure mode 3). That is
  why the raw rating/text/barriers are always shown beside it.

## 6. Mapping to the failure-mode analysis

| Failure mode (`docs/failure-mode-analysis.md`) | Automated coverage |
|---|---|
| 1 — Rating drop with no barrier text | `TestRatingDropMismatch`, `test_blank_barriers_still_allow_rating_drop_to_flag` |
| 2 — Cultural / linguistic variation | None possible with unit tests; see "Non-English barrier text" above |
| 3 — SLM unavailable or hallucinating | Unavailability: `TestOllamaBoundaries` (full). Hallucination: not testable here |
| 4 — False positives / alert fatigue | `TestSeverityTiers` (triage tiers); the false-positive *rate* is measured in `experiments/`, not asserted |
