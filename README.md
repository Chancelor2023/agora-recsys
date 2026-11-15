# Agora
Personalised news recommendation with diversity-aware reranking using the IDEA dataset.

## Current status
The evaluation V1 supports mock predictions with NDCG@10 and Recall@20. The SCRUF-D-inspired reranking V1 runs on synthetic scores. Data processing, relevance scoring and API implementation are still to be built. The synthetic example is not an IDEA record or an experimental result.

## Team and responsibilities
| Member | Main responsibility | Code directory |
| --- | --- | --- |
| Jerry Cheng | Evaluation harness and model-specific preprocessing | `src/agora/evaluation/` |
| Brett Fajgenbaum | Data pipeline, storage and interaction capture | `src/agora/data/` |
| Ronald Donfack | User profiles, relevance model, training and prediction | `src/agora/profiles/`, `src/agora/models/` |
| Romain Cortyl | SCRUF-D-inspired reranking | `src/agora/reranking/` |
| Mathis Groult | API, frontend and integration | `src/agora/api/`, `src/agora/frontend/` |

## Setup
Use Python 3.11 or newer. From the repository root, on Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import agora; print(agora.__version__)"
```
If Python 3.11 is unavailable, use an installed newer version. Activation is optional; the commands above use the environment directly.

The project initially has no runtime dependencies. Add the libraries actually used by your component to `pyproject.toml`, coordinate shared changes, and document their setup. The requirements file installs this local package in editable mode.

## Repository layout
- `src/agora/`: importable application code, organised by component.
- `configs/experiment.toml`: proposed shared experimental settings.
- `data/README.md`: dataset acquisition and storage instructions.
- `examples/`: synthetic shared inputs and separate evaluation labels.
- `docs/interfaces.md`: proposed component interfaces.
- `tests/`: targeted checks to add alongside implementations.
- `reports/week2/`, `reports/week3/`: report preparation and evidence links.
- `artifacts/`: local generated models and embeddings, ignored by Git.

## Dataset
IDEA — Informfully Dataset with Enhanced Attributes.
Source: https://github.com/Informfully/Datasets/tree/main/IDEA
See `data/README.md`. Do not commit raw participant records.

## Development workflow
Read `CONTRIBUTING.md`. Start each task from an up-to-date `main`, work on a branch such as `jerry/evaluation-v1`, and open a pull request with a runnable example.
Agree on `docs/interfaces.md` before integrating the components.

## Running the system and evaluation
Evaluation V1 is available. After setup, run: `python -m agora.evaluation --predictions examples/mock_predictions.json --labels examples/mock_labels.json --output artifacts/evaluation/mock_metrics.json`. Run verification with `python -m unittest discover -s tests -v`. See `src/agora/evaluation/README.md` for formats and hand-calculated expectations. Reranking V1 is available: `python -m agora.reranking --input examples/reranking_mock_input.json --config configs/reranking_agents.json --output artifacts/reranking/mock_predictions.json` (add `--lambda 0` for the ablation); see `src/agora/reranking/README.md`. Other components are pending. Mock outputs are not trained predictions.

## Research reference
Aird et al. (RecSys 2025), Integrating Individual and Group Fairness for Recommender Systems through Social Choice.
https://doi.org/10.1145/3705328.3748087
