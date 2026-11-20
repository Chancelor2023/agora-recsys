# Week 3 evidence (working draft, 20 November 2026)
Everything below is rebuilt from the full IDEA dataset with two commands:
```
python -m agora.build --idea-dir data/raw/Datasets/IDEA/dataset
python -m agora.experiment --split validation
```
All figures are **validation** figures. The test split has not been evaluated: models, lambda and targets are selected on validation only, then frozen (see "Test protocol" below). Full tables, lambda sweep and reader examples: [experiment_validation.md](experiment_validation.md).

## Models
- **Neural recommender (served by the API), NRMS-style** (`src/agora/models/neural.py`, 43,496 trained parameters):
  - news encoder: TF-IDF+SVD text vector, topic and outlet embeddings;
  - user encoder: multi-head self-attention over the last 50 training reads, then additive attention pooling;
  - Wide & Deep click predictor over the user-article interaction and three reader-specific signals: content similarity, topic affinity, outlet affinity;
  - trained with a listwise softmax loss per feed;
  - a position-bias tower (PAL, Guo et al. 2019) absorbs the display-position effect during training and is dropped at recommendation.
- **Logistic regression** (`src/agora/models/relevance.py`): baseline on content similarity, topic affinity and training popularity.
- **Text encoder:** TF-IDF+SVD fitted only on the 5,596 articles published before the end of training (29 Nov 2023, 24:00 Amsterdam time), then applied to every article. No later text shapes the vocabulary.

Reader-independent signals (freshness, topic and outlet popularity) are left out of the neural model by default. On validation, adding them lowers NDCG@10 (0.392 vs 0.413) and does not make the lists more personal (top-10 overlap 0.82 vs 0.85).

## Q1 Training
![Neural model loss](neural_loss_curves.png)

- **Neural model:**
  - trained on 2,916 training feeds (75,790 candidate rows);
  - validation loss falls from the uniform-ranking baseline 3.258 to 3.204 at epoch 6, then rises slowly;
  - stop reason (`neural_report.json`): validation loss not improved by 1e-4 for 8 epochs, weights of epoch 6 kept;
  - no NaN; runs are deterministic (seed 42).
- **Logistic baseline** (`loss_curves.png`):
  - validation log loss 0.4168, against a prevalence baseline of 0.4247;
  - stop reason (`training_report.json`): converged after 382 epochs, improvement below 1e-5 for 50 epochs; the last epoch is the best and the validation loss never rises.

## Q2 Sanity floor (validation)
1,106 of the 1,620 validation feeds are evaluated. 514 feeds are excluded because their reader read nothing within 24 hours, so they have no positive; the count is reported with every result (`excluded_zero_positive_feeds`).

| Ranking | NDCG@10 | Recall@20 |
| --- | --- | --- |
| Random | 0.3091 | 0.7778 |
| Most popular (article, training reads) | 0.4043 | 0.8040 |
| Topic popularity (training, seeded random tie-break) | 0.3251 | 0.7904 |
| Logistic regression | **0.4225** | **0.8652** |
| Neural (NRMS-style) | 0.4127 | 0.8324 |

Paired bootstrap as declared in `docs/test_protocol.md` (`python -m agora.evaluation.bootstrap`, 2,000 resamples of the 484 readers, a reader's feeds stay together, seed 42), 95% CI:

| Comparison | NDCG@10 | Recall@20 |
| --- | --- | --- |
| Neural − random | +0.104 [+0.087, +0.119] | +0.055 [+0.035, +0.072] |
| Neural − most popular (article) | +0.009 [−0.009, +0.026] | +0.028 [+0.011, +0.046] |
| Neural − topic popularity | +0.088 [+0.073, +0.102] | +0.042 [+0.023, +0.060] |
| Logistic − most popular (article) | +0.018 [+0.000, +0.036] | +0.061 [+0.044, +0.078] |
| Logistic − topic popularity | +0.097 [+0.079, +0.115] | +0.075 [+0.058, +0.094] |
| Neural − logistic | −0.010 [−0.026, +0.006] | **−0.033 [−0.048, −0.018]** |

- Both models clearly beat random and topic popularity.
- **Article-level most popular is a strong floor.** No candidate was read during training, so all its scores are 0 and the order is the tie-break on article IDs, which follow publication time: it ranks the oldest article first. That order alone reaches 0.404 NDCG@10. The neural model does not beat it in NDCG@10 (interval includes 0), only in Recall@20; the logistic baseline beats it narrowly. Why is not established yet (one hypothesis: older articles had more time to be read). It ignores the reader, so it is a floor to beat, not a recommender.
- Neural vs logistic: no difference on NDCG@10; on Recall@20 the **logistic baseline is significantly better**. The neural model is kept for serving as the course's complex-model requirement, not because it wins on validation.

## Sanity check: personalisation
Several readers received the same 26 candidates on the same day (6 candidate sets). The mean top-10 overlap between them shows how personal the lists are (Jaccard index; 1 = everyone gets the same list):

| Ranking | Top-10 overlap between readers |
| --- | --- |
| Random | 0.25 |
| Logistic regression | 0.42 |
| Neural | 0.85 |
| Neural + SCRUF-D | 0.79 |
| Topic popularity | 0.94 |
| Most popular (article) | 1.00 |

**Limitation:** the neural model is much less personal than the logistic baseline. It mostly learned which articles appeal to everyone; the article embeddings carry most of the score, the reader's history little.

For today's articles the API adds a personal candidate-generation stage (closest article of every topic, then closest overall; 26 articles), then ranks it. On 558 articles of 19 Nov 2026, the top-10 overlap between 8 readers was 0.12 (measured before the encoder was refitted on past articles; to re-measure).

Three readers before and after SCRUF-D: [experiment_validation.md](experiment_validation.md).

## Q3 Leakage
- **Temporal split** in local time (Europe/Amsterdam): train 23–29 Nov, validation 30 Nov–2 Dec, test 3–7 Dec 2023.
- **Time zone of the raw timestamps:** UTC+1 throughout (no clock change in the period). `python -m agora.data.verify_timezone --idea-dir data/raw/Datasets/IDEA/dataset` matches each read with the closest article page view (epoch time): 34,763 of the 34,871 matched reads (99.7%) are 1 hour apart once rounded to the hour.
- **Profiles and popularity** use training reads only: histories, topic/outlet affinity, topic popularity and the position propensities. Validation and test profiles are frozen at the end of training.
- **Article text vectors:** the encoder is fitted on articles published during training only (5,596 articles); it never sees validation or test text when building its vocabulary.
- **Validation labels** serve only for loss curves, early stopping, model choice and lambda.
- **Test labels** are never read by a model; see the test protocol below.
- **Build checks** (`data/processed/cleaning_report.json`): the build stops if a candidate or a read is missing from the article catalogue (0 of each), or if two feeds share an identifier. It also reports reads outside the experiment (0), invalid publication dates (0), missing values (4 articles without a lead) and eligible readers without a feed per split (0, 0, 27 in test).
- **Topics:** IDEA's 300+ categories are mapped to 12 topics by fixed rules (`src/agora/data/idea.py`); Sky News' "home" section is front-page news (uk_news).

## Q4 Innovation proof (validation, neural model)
Own metric: position-weighted top-10 exposure of the protected topic (politics, agent `public_affairs`), compared with its share of the candidates (8.2%). Ablation: lambda = 0.

| | NDCG@10 | Recall@20 | Politics top-10 exposure |
| --- | --- | --- | --- |
| Ablation (lambda 0) | 0.4127 | 0.8324 | 0.056 |
| SCRUF-D (lambda 0.25) | 0.4048 | 0.8323 | **0.089** (target 0.082 reached) |

- Cost of the reranking: NDCG@10 −0.008, 95% CI [−0.011, −0.006] (same bootstrap as Q2); Recall@20 unchanged.
- Lambda sweep: lambda 0.1 already reaches the target (0.086); 0.25, 0.5 and 1 give 0.089–0.090. The rule for the final lambda is a team decision to take before the freeze.
- The `environment` agent never acts on validation: the neural model already gives environment 0.140 of top-10 exposure, above its 0.048 target.
- The logistic baseline gives politics only 0.053 of top-10 exposure.

![Trade-off](tradeoff_validation.png)

## Q5 Live system
- **API:** `GET /mode` reports the NRMS model and SCRUF-D (`is_mock: false`). `/recommend` takes about 60 ms per IDEA feed; `/recommend_live` takes about 25 ms on today's articles. Both are well under 2 s.
- **Recent articles:**
  - `python -m agora.data.live` fetched 558 articles from the RSS feeds of IDEA's 6 outlets on 19 Nov 2026 (`configs/live_sources.json`);
  - the model scores them with the saved text encoder, without retraining, after the personal candidate-generation stage; SCRUF-D then reranks them;
  - the interface offers "Today's news" next to the IDEA feeds.
- **Interaction logging:** Streamlit interface → API → SQLite (writes serialised by a lock). Displays and clicks are stored and listed with `GET /interactions`.
- To add: the two interface screenshots (today's news list; stored click).

## Test protocol
Full protocol: [docs/test_protocol.md](../../docs/test_protocol.md); leakage checklist: [q3_leakage_checklist.md](q3_leakage_checklist.md).

1. After the team decisions, freeze: `python -m agora.freeze create --note "..."` writes `artifacts/freeze/manifest.json`. It contains the hashes of the models, the embeddings and text encoder, the processed data and labels, both configs and the source code, plus lambda, agents, the git and IDEA commits and the package versions.
2. `python -m agora.freeze verify` lists every difference with the manifest.
3. `python -m agora.experiment --split test` runs only if `verify` passes. It evaluates exactly the six rows fixed in `docs/test_protocol.md` (section 3), with lambda 0 (ablation) and the frozen lambda, and refuses `--lambdas`. Every run is logged in `artifacts/freeze/test_runs.jsonl`. A second run needs `--rerun-reason` and writes `experiment_test_run2`, never over the first; once a run is logged, the manifest can no longer be replaced.

## Annex: position-reweighted NDCG
Readers mostly read what the app displayed at the top: IDEA's original display order alone scores 0.508 NDCG@10. Position-reweighted NDCG@10 (labelled IPS-NDCG@10 in the generated tables) weights each read by 1 / P(seen | position), with propensities estimated on the training split. These propensities come from the old display order, which itself followed a ranking, so the correction is only partial. No decision in this report relies on it.

| Ranking | Position-reweighted NDCG@10 |
| --- | --- |
| Random | 0.2713 |
| Most popular (article) | 0.3066 |
| Topic popularity | 0.3049 |
| Logistic regression | 0.3647 |
| Neural (NRMS-style) | 0.3674 |
| Neural + SCRUF-D (lambda 0.25) | 0.3602 |

## Pending
- Team decisions before the freeze: final lambda rule, rows of the test table, neural vs logistic as the main model (D1–D9).
- Q6: clean-machine launch by a member who did not write the code, timed, following the README only.
- Re-measure the live top-10 overlap with the refitted encoder.
- Test split: freeze, then run `python -m agora.experiment --split test` once.
- Q7 and the global risk level: team.
