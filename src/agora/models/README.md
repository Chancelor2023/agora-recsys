# models
Owner: Ronald.

Content representations, relevance scoring, training and prediction.

## Model
Logistic regression over three point-in-time features per (reader, candidate): content similarity to the reader's training-history profile, topic affinity, and training popularity. It is trained epoch by epoch (full-batch Adam, L2 scaled like scikit-learn's `C`). Training and validation log loss are recorded at every epoch; the epoch with the lowest validation loss is kept (early stopping, `--patience`). Validation labels serve only for this curve and model selection; test labels are never read.

## Commands
```
python -m agora.models encode --articles <articles> --output artifacts/models/embeddings.npz
python -m agora.models train --config configs/experiment.toml --articles <articles> --reads <reads_train.csv> --embeddings artifacts/models/embeddings.npz --model artifacts/models/model.json --feeds <feeds_train.jsonl> --labels <train_labels.json> --validation-feeds <feeds_validation.jsonl> --validation-labels <validation_labels.json> --report artifacts/models/training_report.json --curves reports/week3/loss_curves.png
python -m agora.models predict ... --split validation --output <predictions.json>   # --ranking model|popularity|random|content
python -m agora.models freeze --model artifacts/models/model.json --validation-note "<selected settings>"
python -m agora.models predict ... --split test --output <predictions.json>         # requires the freeze
```
`training_report.json` holds the full per-epoch history, the best epoch and the prevalence-baseline log loss.
Synthetic end-to-end check: `python examples/smoke.py` (writes `artifacts/ronald_smoke/loss_curves.png`).
