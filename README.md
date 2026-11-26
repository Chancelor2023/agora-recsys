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

## Run the system locally (target: under 10 minutes)

Python 3.11 or newer and Git are required. Commands are run from the repository root.
This path serves the **frozen** models, the ones evaluated once on the test split: they are downloaded (under 100 MB) instead of being retrained, so the IDEA download and the build are not needed. The first clean-machine test took 27 minutes because it rebuilt everything (`reports/week3/q6_clean_machine.md`); rebuilding is now optional (see below).

**1. Clone and create the environment**

Windows, PowerShell or CMD:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
python -m venv .venv
.venv\Scripts\activate
python --version
```

Windows, Git Bash:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
python -m venv .venv
source .venv/Scripts/activate
python --version
```

macOS / Linux:
```
git clone https://github.com/Chancelor2023/agora-recsys.git
cd agora-recsys
python3 -m venv .venv
source .venv/bin/activate
python --version
```
`python --version` must print 3.11 or later, and the prompt must start with `(.venv)`. On Windows, use `python` (or `py -3.12`, `py -3.13`… for a version you have installed), not `python3`, which may open the Microsoft Store shortcut; the environment's scripts are in `.venv/Scripts`, not `.venv/bin`. If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that window first.

**2. Install the dependencies** (the longest step: PyTorch is about 200 MB)
```
python -m pip install uv
uv pip install -r requirements.txt
```
`uv` installs the same packages as pip, with parallel downloads. Without it: `python -m pip install -r requirements.txt` (slower). On Linux, first run `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`, otherwise PyPI's PyTorch brings several GB of GPU libraries.

**3. Download the frozen models and data** (can run in a second terminal while step 2 installs)
```
python scripts/fetch_models.py
```
It downloads the release archive, extracts `data/processed/` and `artifacts/models/`, and checks every file against the freeze manifest (`artifacts/freeze/manifest.json`). It uses the Python standard library only.

**4. Start the API** (keep this terminal open)
```
python -m uvicorn agora.api.main:app
```
Check http://127.0.0.1:8000/mode: it must show `"is_mock": false` and model version `nrms-8dd250d0620b523b+scruf-d-lambda-0.1`. Interactive documentation: http://127.0.0.1:8000/docs

**5. Start the interface** (second terminal, in the `agora-recsys` folder, environment activated as in step 1)
```
python -m streamlit run src/agora/frontend/app.py --server.headless true
```
Open http://localhost:8501 and choose a reader: four demo readers (`configs/demo_personas.json`, invented names for real anonymised IDEA readers) or any other one. Then choose "Today's news" (if fetched) or an archive edition, and a window on the world: "Only my interests" (lambda 0), "Balanced" (the evaluated setting) or "Wide" (lambda 1). The "With and without Agora" tab compares the selection with the ranking by interest alone. "Read" sends a click to `/log_interaction`; the last stored events are shown at the bottom of the page (SQLite file `artifacts/agora.db`). `--server.headless true` skips Streamlit's first-run e-mail prompt.

**Optional: fetch today's articles** (needs internet)
```
python -m agora.data.live
```
Writes `data/live/articles_live.json` from the RSS feeds listed in `configs/live_sources.json`. Run it again to refresh; the API picks up the new file without a restart.

**Optional: rebuild everything from the raw dataset** (110 MB download, then 3 to 10 minutes; 147 s, 327 s and 581 s measured)
```
git clone --depth 1 https://github.com/Informfully/Datasets.git data/raw/Datasets
python -m agora.build --idea-dir data/raw/Datasets/IDEA/dataset
```
This runs the data pipeline (`data/processed/`), encodes the articles, trains the logistic baseline and the neural recommender (`artifacts/models/`), and writes the loss curves to `reports/week3/`. Retrained models are close to the frozen ones but not identical (different model version).

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
| `GET /users/{user_id}/profile` | Topics the reader read during the training period |
| `GET /recommend?user_id=...&feed_id=...` | Scored and reranked IDEA feed, with `latency_ms`; optional `lam` (0 to 2) overrides lambda for this call |
| `GET /live/status`, `GET /recommend_live?user_id=...` | Today's articles, scored for the reader and reranked; optional `lam` as above |
| `POST /log_interaction` | Store a `display` or `click` event in SQLite |
| `GET /interactions?user_id=...` | Last stored events |

A call with `lam` uses its own reranker and exposure memory, so the default (frozen) setting is never changed by it. `AGORA_MODEL=logistic` makes the API serve the logistic model instead of the neural one (default: neural).

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
