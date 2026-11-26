# Agora

Personalised news recommendation with fairness-aware reranking, on the IDEA dataset.

A neural news recommender (NRMS-style: attention over each reader's reading history, trained on IDEA with a position-bias correction) scores articles for a reader. A SCRUF-D-inspired reranker (Aird et al., RecSys 2025) then reorders the list so that public-affairs news keeps a fair share of top-10 exposure. A FastAPI backend serves the result to a Streamlit interface, on IDEA's 2023 daily feeds or on today's articles fetched from the outlets' RSS feeds, and stores reader interactions in SQLite.

## Team and responsibilities

| Member | Main responsibility | Code directory |
| --- | --- | --- |
| Jerry Cheng | Evaluation harness and model-specific preprocessing | `src/agora/evaluation/` |
| Brett Fajgenbaum | Data pipeline, storage and interaction capture | `src/agora/data/` |
| Ronald Donfack | User profiles, relevance model, training and prediction | `src/agora/profiles/`, `src/agora/models/` |
| Romain Cortyl | SCRUF-D-inspired reranking | `src/agora/reranking/` |
| Mathis Groult | API, frontend and integration | `src/agora/api/`, `src/agora/frontend/` |

## Run the system locally (about 25 minutes on a laptop)

Python 3.11 or newer and Git are required. Commands are run from the repository root.
Measured on a Windows laptop with a clean environment (22 Nov 2026): 27 minutes in total, of which
15 for installing the dependencies and 10 for the build (`reports/week3/q6_clean_machine.md`).

**1. Clone and create the environment**

Python 3.11 or later is required. Pick the block for your terminal.

Windows, PowerShell or CMD:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
python -m venv .venv
.venv\Scripts\activate
python --version
python -m pip install -r requirements.txt
```

Windows, Git Bash:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
py -3.11 -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip install -r requirements.txt
```
On Windows, use `py`, not `python3`: `python3` may open the Microsoft Store shortcut and create nothing, without any error. The environment's scripts are in `.venv/Scripts`, not `.venv/bin`.

macOS / Linux:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```
`python --version` must print 3.11 or later, and the prompt must start with `(.venv)`. On Windows, `python -m venv .venv` and `py -3.11 -m venv .venv` both work, but not `python3`, which may open the Microsoft Store shortcut. If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window first.

**2. Download the IDEA dataset** (about 110 MB, public)
```
git clone --depth 1 https://github.com/Informfully/Datasets.git data/raw/Datasets
```

**3. Build the data, embeddings and models** (3 to 10 minutes depending on the machine; 147 s, 327 s and 581 s measured)
```
python -m agora.build --idea-dir data/raw/Datasets/IDEA/dataset
```
This runs the data pipeline (`data/processed/`), encodes the articles, trains the logistic baseline and the neural recommender (`artifacts/models/`), and writes the loss curves to `reports/week3/`.

**Optional: fetch today's articles** (needs internet)
```
python -m agora.data.live
```
Writes `data/live/articles_live.json` from the RSS feeds listed in `configs/live_sources.json`. Run it again to refresh; the API picks up the new file without a restart.

**4. Start the API** (keep this terminal open)
```
python -m uvicorn agora.api.main:app
```
Check http://127.0.0.1:8000/mode: it must show `"is_mock": false`. Interactive documentation: http://127.0.0.1:8000/docs

**5. Start the interface** (second terminal, in the `agora-recsys` folder, environment activated as in step 1: `.venv\Scripts\activate`, `source .venv/Scripts/activate` in Git Bash, `source .venv/bin/activate` on macOS / Linux)
```
python -m streamlit run src/agora/frontend/app.py
```
Open http://localhost:8501, choose a reader, then "Today's news" (if fetched) or one of their IDEA daily feeds. "Read" sends a click to `/log_interaction`; the last stored events are shown at the bottom of the page (SQLite file `artifacts/agora.db`).

**Checks and test protocol** (optional)
```
python -m agora.data.verify_timezone --idea-dir data/raw/Datasets/IDEA/dataset
python -m agora.experiment --split validation
python -m agora.freeze create --note "why this state is frozen"
python -m agora.freeze verify
python -m agora.experiment --split test
```
`verify_timezone` checks that IDEA's text timestamps are UTC+1. The validation experiment writes `reports/week3/experiment_validation.*`. The test split runs only after `freeze create`: the manifest is the only freeze mechanism. All checks (manifest, `--lambdas` refused, earlier run) pass before the run is logged in `artifacts/freeze/test_runs.jsonl`, so a refused attempt does not count as a run; a second run needs `--rerun-reason`. The run also computes the declared bootstrap comparisons, so test is read once (details: `docs/test_protocol.md`).

## API endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `GET /mode` | Real or mock mode, model version |
| `GET /users`, `GET /users/{user_id}/feeds` | Readers and their validation/test daily feeds |
| `GET /recommend?user_id=...&feed_id=...` | Scored and reranked IDEA feed, with `latency_ms` |
| `GET /live/status`, `GET /recommend_live?user_id=...` | Today's articles, scored for the reader and reranked |
| `POST /log_interaction` | Store a `display` or `click` event in SQLite |
| `GET /interactions?user_id=...` | Last stored events |

Without `data/processed/` and `artifacts/models/` (or with `AGORA_MOCK=1`), the API falls back to the synthetic feed in `examples/feed_input.json` (`is_mock: true`).

## Tests

```
python -m pytest
```

## Reproduce the results

```
python -m agora.experiment --split validation
```
Writes `reports/week3/experiment_validation.md` (random, most-popular, logistic, neural and neural + SCRUF-D with NDCG@10, Recall@20, position-debiased IPS-NDCG@10 and topic exposure; lambda sweep; three reader examples) and the trade-off plot. The test split is run once, after `python -m agora.freeze create` (see "Checks and test protocol" above): `--split test`.

## Retrain or re-run single components

- Data pipeline: `python -m agora.data.pipeline --idea-dir data/raw/Datasets/IDEA/dataset --out data/processed` (see `src/agora/data/README.md`).
- Models: `python -m agora.models train-neural ...` (neural), `train ...` (logistic), `predict` on train or validation (see `src/agora/models/README.md`).
- Reranking: `python -m agora.reranking --input <predictions.json> --articles data/processed/articles.json --output <reranked.json>`; `--lambda 0` is the ablation (see `src/agora/reranking/README.md`).
- Evaluation: `python -m agora.evaluation --config configs/experiment.toml --predictions <predictions.json> --labels data/processed/validation_labels.json` (see `src/agora/evaluation/README.md`).

Validation labels are used for model selection only. Test predictions come only from `python -m agora.experiment --split test` after `python -m agora.freeze create`, and the test split is evaluated once.

## Repository layout

- `src/agora/`: application code by component; `src/agora/build.py` rebuilds every local artifact.
- `configs/experiment.toml`: shared temporal split, timezone and evaluation settings.
- `configs/reranking_agents.json`: fairness agents, targets and lambda.
- `data/README.md`: dataset acquisition, timestamps and pipeline outputs.
- `docs/`: component interfaces and the temporal protocol.
- `examples/`: synthetic inputs used by tests and smoke runs.
- `reports/week2/`, `reports/week3/`: report evidence.
- `data/raw/`, `data/processed/`, `artifacts/`: local data and generated files, ignored by Git.

## Dataset

IDEA — Informfully Dataset with Enhanced Attributes. Source: https://github.com/Informfully/Datasets/tree/main/IDEA. Do not commit raw participant records.

## Development workflow

Read `CONTRIBUTING.md`. Start each task from an up-to-date `main`, work on a branch, and open a pull request with a runnable example.

## Research reference

Aird et al. (RecSys 2025), Integrating Individual and Group Fairness for Recommender Systems through Social Choice. https://doi.org/10.1145/3705328.3748087
