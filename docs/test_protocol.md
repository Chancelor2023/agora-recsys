# Final test protocol (Week 3)

Status: **proposed by Jerry; team decisions taken on 22 Nov 2026** (sections 3 and 6),
**signed off by the five members on 22 Nov 2026** (section 10). Once signed off, this file is
frozen with the experiment and changes only through a dated amendment that
explains why.

It complements [temporal_protocol.md](temporal_protocol.md) (splits, profiles,
labels), which remains the reference for data preparation.

## 1. Principle

The test split answers one pre-declared question with frozen settings. Everything
that can change a test number is chosen on train/validation, written down here,
hashed in the freeze manifest, and checked again just before the run.
The test split is evaluated **once**. No setting is changed after test numbers
have been seen.

## 2. What may use which data

| Quantity | Allowed data |
| --- | --- |
| Profiles, popularity, affinities, IPS propensities | Training reads only (validation/test profiles frozen at `train_end`) |
| Model weights | Training feeds and labels |
| Early stopping, model choice, lambda, targets | Validation labels |
| Text encoder (TF-IDF vocabulary, IDF, SVD basis) | Articles **published before `train_end`** (`published_at < 2023-11-29T23:00Z`) |
| Encoding of validation, test and RSS articles | `transform` with the frozen encoder; no refit |
| Article metadata at serving time (topic, outlet, publication time) | Allowed for a feed when the article is a candidate of that feed |
| Topic mapping rules (`src/agora/data/idea.py`) | Frozen as committed; written with all IDEA categories in view (see Q3 checklist) |
| Test labels | Final evaluation only |

Interactions of the validation or test period are never used as features, even
when they happen before a feed (no "online popularity").

## 3. Rankings reported on test (closed list, decided 22 Nov 2026)

Closed list, all evaluated on the same feeds, candidates and labels:

1. Random (seed 42).
2. Most popular, article level (training reads). Reported with its explanation:
   no test candidate was read during training, so every score is 0 and the
   order comes from the tie-break on article IDs. These IDs follow publication
   time (rank correlation 1.0 on validation candidates), so this ranking is
   "oldest first".
3. Topic popularity (training reads), **with a seeded random tie-break**: our
   declared adaptation of most-popular to new articles.
4. Logistic regression.
5. Neural (NRMS-style), which is also the lambda = 0 ablation.
6. Primary model + SCRUF-D at the frozen lambda.

No other ranking, lambda or variant is computed on test.

## 4. Metrics

- Primary: NDCG@10. Secondary: Recall@20 (about 26 candidates per feed, so
  Recall@20 is close to saturation; it is reported but not used alone).
- Macro average by feed. Feeds without any positive label are **counted and
  excluded** from both averages (`zero_positive_policy = exclude_and_report_count`).
  `total_feeds`, `evaluated_feeds` and `excluded_zero_positive_feeds` are reported
  for every ranking and must be identical across rankings. Interpretation: ranking
  quality on reader-days with at least one observed read.
- Innovation metric: position-weighted top-10 exposure of each protected group,
  compared with that group's share of the **test** candidates (an input statistic,
  computed without labels).
- Supplementary (kept, section 6): NDCG@5 / Recall@5 / top-5 exposure for public affairs,
  as announced in the temporal protocol.
- Exploratory only (appendix, no decision based on it): position-reweighted NDCG
  (`evaluation/ips.py`). Its propensities are read rates per position, so they mix
  the position effect with the relevance of the old ranking.

## 5. Declared comparisons and uncertainty

Paired bootstrap (`agora.evaluation.bootstrap`), resampling **readers**
(unit user), 2,000 resamples, seed 42, 95 % percentile interval, on NDCG@10
and Recall@20. On test, the comparisons are computed by the test run itself
(`comparisons` in `experiment_test.json`, table in `experiment_test.md`), so the
test split is read once; `python -m agora.evaluation.bootstrap` compares
prediction files (validation):

1. Primary model vs random, vs most popular (article), vs topic popularity (Q2).
2. Neural vs logistic regression.
3. Primary model + SCRUF-D vs primary model (lambda = 0): accuracy cost (Q4),
   reported next to the exposure change.

## 6. Choices made on validation, before the freeze

### Lambda rule (decided 22 Nov 2026)
The **smallest lambda of the grid {0, 0.1, 0.25, 0.5, 1}** whose validation top-10
exposure of the `public_affairs` group reaches its share of validation candidates.
Applied on the validation run regenerated after the encoder fix: lambda 0.1 gives
0.086 against a target of 0.082 (0 gives 0.056), so **lambda = 0.1**.

### Primary model (decided 22 Nov 2026)
**Neural (NRMS-style).** On validation (encoder fitted on training articles only,
paired bootstrap over 484 readers) the two models do not differ in NDCG@10, the
primary metric (neural - logistic: -0.010, 95 % CI [-0.026, +0.006]). Logistic
regression is better in Recall@20 (-0.033 [-0.048, -0.018]), a secondary metric
not used alone (section 4), and gives more personal lists. With no difference on
the primary metric, the team keeps the neural model for the course requirement of
a complex model; this is stated in the report as a choice, not as a measured gain.
Logistic regression stays in the test table.

### Supplementary metrics (decided 22 Nov 2026)
Kept: NDCG@5, Recall@5 and top-5 exposure for public affairs, where only reads of
the `public_affairs` topics (politics) count as relevant and feeds without such a
read are excluded and counted. Reported for every ranking; no decision is based on
them. The `environment` agent is kept although it does not act on validation
(exposure already above its target); this is disclosed.

### Fixed values (filled at the freeze)
| Item | Value |
| --- | --- |
| Primary model | Neural (NRMS-style) |
| Lambda | 0.1 |
| Agents and targets | public_affairs (politics) 0.082; environment 0.048 (`configs/reranking_agents.json`) |
| Git commit of the frozen code | |
| IDEA dataset commit | `9a71a3e439daf3e0efc933b1927907235afa0575` (proposed) |
| Freeze manifest SHA-256 | |

## 7. Freeze manifest

The freeze records SHA-256 hashes of everything that determines a test number,
and the test command refuses to run if any of them changed:

- models: `model.json`, `neural.pt`;
- text representation: `embeddings.npz`, `embeddings.npz.json`,
  `embeddings.npz.encoder.joblib`;
- data inputs: `articles.json`, `reads_train.csv`, `eligible_users.json`,
  `feeds_{train,validation,test}.jsonl`, `{train,validation,test}_labels.json`;
- settings: `configs/experiment.toml`, `configs/reranking_agents.json` (lambda,
  agents, targets, allocation, window, seed);
- code: one hash over `src/agora/**/*.py`;
- environment: versions of Python, numpy, pandas, scikit-learn, torch, joblib and
  fastapi; git commit (the freeze warns if tracked files differ) and IDEA commit
  are recorded;
- the closed list of rankings of section 3 and the frozen lambda.

`python -m agora.freeze create` writes `artifacts/freeze/manifest.json`; it is the
only freeze mechanism (the former `agora.models freeze` sidecars are removed).
`python -m agora.experiment --split test` runs every check (manifest, `--lambdas`
refused, earlier run) before the run is logged, so a refused attempt is not a run.

The frozen **artifacts** are used for the test. They are not rebuilt: a rebuild
on another machine gives a different neural model hash.

`tests/test_freeze_protocol.py` checks these refusals on synthetic data.

## 8. Run procedure

The artifacts are frozen without retraining and are git-ignored, so the freeze and
the test run happen on the machine that holds them (Romain's).

1. Check out the merged commit on a clean tree.
2. `python -m agora.freeze create --note "..."`, then `python -m agora.freeze verify`.
3. Commit and push the manifest at once (`git add -f artifacts/freeze/manifest.json`),
   **before** the test run, so the freeze is timestamped before any test number exists.
4. `python -m agora.experiment --split test`, once. It evaluates only the rankings of
   section 3, computes the comparisons of section 5 and refuses to overwrite an
   existing `experiment_test.json`.
5. Commit `reports/week3/experiment_test.*` and the run log
   (`git add -f artifacts/freeze/test_runs.jsonl`: start, finish, manifest and
   result hashes); add the operator and the exit status in the PR.
6. Copy the numbers to the report without editing them.

## 9. Re-running

| Allowed: re-run with the same manifest | Not allowed |
| --- | --- |
| The command failed before producing results (crash, wrong path, manifest refused for a technical reason) | Changing lambda, model, features, topics, epochs, seeds or targets after seeing test numbers |
| A bug is found **in the evaluation code**: fix it, re-run, publish both runs and explain | Re-running until a number improves, or keeping only the best run |
| Re-run to check determinism: the numbers must be identical | Adding or removing rankings or metrics after seeing test numbers |

Every run, including failed ones, is logged.

## 10. Sign-off

| Member | Agreed (date) |
| --- | --- |
| Jerry Cheng | 2026-11-22, internal team discussion |
| Brett Fajgenbaum | 2026-11-22, internal team discussion |
| Ronald Donfack | 2026-11-22, internal team discussion |
| Romain Cortyl | 2026-11-22, internal team discussion |
| Mathis Groult | 2026-11-22, internal team discussion |
