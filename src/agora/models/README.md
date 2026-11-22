# models
Owner: Ronald.

Content representations, relevance models, training and prediction.

## Neural recommender (served by the API) — `neural.py`
NRMS-style news recommender (Wu et al., EMNLP 2019), in PyTorch:
- **News encoder:** the article's text vector (`embeddings.py`), projected, plus learned topic and outlet embeddings.
- **User encoder:** multi-head self-attention over the reader's last 50 reads before the feed, then additive attention pooling into one user vector.
- **Click predictor (Wide & Deep):** a linear part over the reader-specific signals (content similarity, topic affinity, outlet affinity) plus an MLP over the user-article interaction. The deep part starts at zero and only adds what improves validation loss. `--with-global-signals` also feeds freshness and topic/outlet popularity; this lowered validation NDCG@10 (0.389 vs 0.405).
- **Position-bias tower (PAL, Guo et al., RecSys 2019):** in IDEA readers mostly read what was displayed at the top. A learned per-position term is added to the logit in training only and dropped at recommendation, so the network learns relevance rather than the old display order.
- **Training:** listwise softmax loss per feed (as NRMS), AdamW, epoch by epoch, early stopping on validation loss; seed 42.
- **New articles:** it scores articles from their text, topic and outlet, so today's RSS articles can be recommended without retraining (`live.py`, `embeddings.load_text_encoder`).

Article-level popularity is not given to the network: none of the validation or test candidates was read during training (they are new articles).

Known limitation: lists are much less personal than the logistic baseline's (top-10 overlap between readers with the same candidates 0.82 vs 0.49, `python -m agora.experiment`).

## Logistic baseline — `relevance.py`
Logistic regression on content similarity, topic affinity and article popularity, trained epoch by epoch with early stopping.

## Text vectors — `embeddings.py`
TF-IDF (1–2 grams) reduced to 256 dimensions by truncated SVD (default, offline). With `--config`, as in `agora.build`, TF-IDF and SVD are fitted only on articles published before `train_end` (5,596 articles on IDEA), then applied to all articles, so validation and test text never shapes the vocabulary. The fitted encoder is saved (`embeddings.npz.encoder.joblib`) to encode new articles identically. `--backend sentence-transformer` uses `all-MiniLM-L6-v2` instead (needs Hugging Face access).

## Commands
```
python -m agora.models encode --articles <articles> --output artifacts/models/embeddings.npz
python -m agora.models train-neural --config configs/experiment.toml --articles <articles> --reads <reads_train.csv> --embeddings artifacts/models/embeddings.npz --model artifacts/models/neural.pt --feeds <feeds_train.jsonl> --labels <train_labels.json> --validation-feeds <feeds_validation.jsonl> --validation-labels <validation_labels.json> --report artifacts/models/neural_report.json --curves reports/week3/neural_loss_curves.png
python -m agora.models train ... --model artifacts/models/model.json ...     # logistic baseline, same options
python -m agora.models predict ... --model <model.json|neural.pt> --split validation --output <predictions.json>
```
`predict --split test` is refused: test predictions come only from the declared test run (`python -m agora.freeze create`, then `python -m agora.experiment --split test`, see `docs/test_protocol.md`).
`--no-position-debias` trains the neural model without the position tower (ablation). `python -m agora.build` runs encode and both trainings. Synthetic end-to-end check: `python examples/smoke.py`.
