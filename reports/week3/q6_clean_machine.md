# Q6 clean-machine launch (22 November 2026)

**Result: No.** Following `README.md` only, the system was running with a stored click after
**27 min 09 s**, for a 10-minute target. The system itself works end to end with no code change.

## Setup
- **Tester:** Jerry Cheng. He wrote the evaluation code (harness, bootstrap, protocol tests), not
  the data pipeline, the models, the API, the interface or the README.
- **Machine:** Windows 11 Home laptop, PowerShell 5.1, Python 3.13 (Microsoft Store), Git 2.53.
- **Clean conditions:** new folder `C:\q6`, fresh clone, new virtual environment, pip cache
  disabled (`PIP_NO_CACHE_DIR=1` in the same window), no data or model reused. Every dependency
  was downloaded and installed (no "already satisfied" package).
- **Commit tested:** `3fe86ca5f1e83c187ccf075b89ac9713b2275440` (`main` after PR #11).
- **Not run:** the optional RSS step and the "Checks and test protocol" section (the test split
  had already been evaluated once under the freeze).

## Timeline
| Step (README) | Start → end | Duration |
| --- | --- | --- |
| 1. Clone, virtual environment, `pip install -r requirements.txt` | 20:32:04 → 20:46:52 | 14 min 48 s |
| 2. Download IDEA | 20:46:52 → 20:46:56 | 0 min 04 s |
| 3. `python -m agora.build` | 20:46:56 → 20:56:37 | 9 min 41 s |
| 4–5. API, interface, one feed ranked, click stored | 20:56:37 → 20:59:13 | 2 min 36 s |
| **Total** | 20:32:04 → 20:59:13 | **27 min 09 s** |

Build steps (console output): data pipeline 34.0 s, article embeddings 89.7 s, logistic training
121.7 s, neural training 335.0 s.

## System check
- `GET /mode`: `{"is_mock": false, "model_version": "nrms-effc04c1c84b71ac+scruf-d-lambda-0.1",
  "users": 558, "feeds": 4276}` ([q6_mode.png](q6_mode.png)).
- Interface: reader `23g7LwZEmKdFLMM4k`, IDEA feed of 30 Nov 2023 ranked; ten `display` events
  and one `click` stored in SQLite and listed under "Last interactions stored in the database"
  ([q6_click.png](q6_click.png)).
- API log: `GET /recommend` 200, then `POST /log_interaction` 200 for every display and the
  click, and `GET /interactions` 200.
- The clean build retrains the models, so the served model (`nrms-effc04c1…`) is not the frozen
  one evaluated on test (`nrms-8dd250d0…`).

## README gaps
1. `py -3.11 -m venv .venv` fails when the Windows `py` launcher is not installed (Python from the
   Microsoft Store). `python -m venv .venv` works. Fixed in the README after this run.
2. The README announced "about 10 minutes"; the measured time is 27 minutes on this laptop.
   Updated in the README after this run.

## Why it takes longer than 10 minutes
- **Installation (14 min 48 s):** about 300 MB were downloaded at about 7 MB/s, so under one
  minute; most of the time went into installing the files (PyTorch, transformers, pyarrow, scipy).
  Antivirus scanning of each file is a likely cause, not verified.
- **Build (9 min 41 s):** a clean launch retrains both models; neural training alone takes 5.6 min.

## Proposed fixes for the showcase (to be re-timed)
- Make `sentence-transformers` truly optional in `requirements.txt` (it pulls transformers,
  tokenizers and huggingface-hub but is not used by default).
- Offer the frozen models as a download (checked against the hashes of
  `artifacts/freeze/manifest.json`), so that a clean launch only runs the data pipeline and serves
  the models evaluated on test.

## Earlier attempt (discarded)
A first attempt the same evening (19:58) was not valid: after `py -3.11` failed, no virtual
environment was created, so pip installed into the tester's existing Python, where most packages
were already present. It reached the end of the build after 17 min 00 s, already over the target.
Its folder and the packages it had installed were removed before the clean run above.

## Evidence files
- [q6_transcript.txt](q6_transcript.txt): PowerShell transcript with the four timestamps (identity
  lines removed; PowerShell 5.1 does not record the output of pip, the build or uvicorn inside the
  virtual environment).
- [q6_mode.png](q6_mode.png), [q6_click.png](q6_click.png): screenshots taken during the run.
