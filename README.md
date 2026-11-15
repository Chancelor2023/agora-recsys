
# Agora

Personalised news recommendation with diversity-aware reranking using the IDEA dataset.

## Current status

The repository contains the initial component scaffold and a first mock recommendation API.

The API currently returns synthetic recommendations based on `examples/feed_input.json`. These are integration examples, not IDEA records or experimental results. Data processing, relevance scoring, SCRUF-D-inspired reranking and evaluation are still under development.

## Team and responsibilities

| Member | Main responsibility | Code directory |
| --- | --- | --- |
| Jerry Cheng | Evaluation harness and model-specific preprocessing | `src/agora/evaluation/` |
| Brett Fajgenbaum | Data pipeline, storage and interaction capture | `src/agora/data/` |
| Ronald Donfack | User profiles, relevance model, training and prediction | `src/agora/profiles/`, `src/agora/models/` |
| Romain Cortyl | SCRUF-D-inspired reranking | `src/agora/reranking/` |
| Mathis Groult | API, frontend and integration | `src/agora/api/`, `src/agora/frontend/` |

## Setup

Use Python 3.11 or newer.

From the repository root, on Windows CMD:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.11 is unavailable, use an installed newer version and replace `py -3.11` with the appropriate Python command.

## Running the recommendation API

Start the development server from the repository root:

```cmd
.venv\Scripts\python.exe -m uvicorn agora.api.main:app --reload
```

Available endpoints:

- `GET /health`: API health check.
- `GET /recommend?user_id=demo-user&feed_id=demo-feed`: returns synthetic recommendations.

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

The recommendation endpoint currently uses the synthetic candidates from `examples/feed_input.json`. It sets `is_mock` to `true` and uses the model version `mock-v1`.

## Running tests

From the repository root:

```cmd
.venv\Scripts\python.exe -m pytest
```

## Repository layout

- `src/agora/`: importable application code, organised by component.
- `configs/experiment.toml`: proposed shared experimental settings.
- `data/README.md`: dataset acquisition and storage instructions.
- `examples/`: synthetic shared inputs and separate evaluation labels.
- `docs/interfaces.md`: proposed component interfaces.
- `tests/`: targeted checks for implemented components.
- `reports/week2/`, `reports/week3/`: report preparation and evidence links.
- `artifacts/`: local generated models and embeddings, ignored by Git.

## Dataset

IDEA — Informfully Dataset with Enhanced Attributes.

Source: https://github.com/Informfully/Datasets/tree/main/IDEA

See `data/README.md`. Do not commit raw participant records.

## Development workflow

Read `CONTRIBUTING.md`. Start each task from an up-to-date `main`, work on a branch such as `jerry/evaluation-v1`, and open a pull request with a runnable example.

Agree on `docs/interfaces.md` before integrating the components.

## Research reference

Aird et al. (RecSys 2025), Integrating Individual and Group Fairness for Recommender Systems through Social Choice.

https://doi.org/10.1145/3705328.3748087
