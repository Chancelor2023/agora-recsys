# Q3 checklist: temporal separation and leakage

Owner: Jerry (with Brett for data). Status: **working draft, not yet signed off.**
Checked on commit `b2f6342` (PR #6), full IDEA rebuild in a clean virtual
environment (Python 3.13, Windows), IDEA commit `9a71a3e`. Rows 2, 5, 7, 14, 15 and
19–23 updated on 22 Nov 2026 after the fixes (PR #6 and the freeze-preparation branch). **No test-split
evaluation was run.** Final values are filled at the freeze
([docs/test_protocol.md](../../docs/test_protocol.md)).

Legend: **OK** verified and reproduced · **FIX** must change before the freeze ·
**DISCLOSE** acceptable if stated in the report · **TODO** not checked yet.

## Splits and timestamps

| # | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| 1 | Local half-open splits: train 23-29 Nov, validation 30 Nov-2 Dec, test 3-7 Dec 2023 | `cleaning_report.json` boundaries 22 Nov 23:00 UTC, 29 Nov 23:00 UTC, 2 Dec 23:00 UTC, 7 Dec 23:00 UTC; `tests/test_pipeline.py` (PR #7) | OK |
| 2 | IDEA text timestamps are Europe/Amsterdam (UTC+1) | `python -m agora.data.verify_timezone`: 34,763 of the 34,871 reads matched to an article view in `views.json` are +1 h once rounded to the hour (99.7%) | OK |
| 3 | Feed and read timestamps use the same clock | 30,169 reads happen after the reader was first served the article, 889 before | OK |
| 4 | Counts reproduce | 558 readers; 3,317 / 1,620 / 2,656 feeds; 0 candidates or reads missing from the catalogue | OK |
| 5 | Build fails when a candidate is missing from the catalogue | `pipeline.py` stops before any export if a candidate or a read is missing (`tests/test_data_checks.py`) | OK |

## Labels

| # | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| 6 | First read in [feed, min(next feed, +24 h, split end, data end)) | `pipeline.py` `add_windows`, `build_labels`; window and boundary tests in PR #7 | OK |
| 7 | Labels never mix readers | Merge on (user, article); `tests/test_data_checks.py::test_labels_never_use_another_readers_read` puts another reader's read inside the window, so an article-only merge fails it | OK |
| 8 | Labels exported separately from model inputs | `feeds_*.jsonl` have no label; the harness rejects predictions that contain `label` | OK |
| 9 | Next-feed bound from the unfiltered feed sequence | Computed after filtering; no feed is removed on IDEA, so no effect here (differs only with `--sample-feeds`) | DISCLOSE |

## Profiles and features

| # | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| 10 | Validation/test profiles frozen at `train_end` | `TemporalProtocol.cutoff` returns `train_end` outside train (`profiles/history.py` L33-43) | OK |
| 11 | Training-feed history strictly before the feed | Strict `ts < cutoff` (`history.py` L75, `neural.py` L124) | OK |
| 12 | `reads_train.csv` contains only training reads | `ProfileBuilder` raises otherwise (`history.py` L62-63) | OK |
| 13 | Popularity and IPS propensities from training only | `_group_popularity(builder, train_end)`; `estimate_propensity(train feeds, train labels)` | OK |
| 14 | Eligibility: at least 5 distinct articles read in train | `pipeline.py` and `ProfileBuilder.eligible`; `tests/test_data_checks.py` checks that a reader with 4 training articles gets no exported feed | OK |
| 15 | Text encoder fitted only on articles available before `train_end` | `encode --config` fits TF-IDF/SVD on the 5,596 articles published before `train_end` (`embeddings.npz.json`: `fit_before`, `fit_articles`); `tests/test_freeze.py` checks a future-only word is not in the vocabulary | OK |
| 16 | Topic mapping independent of future data | Rules written with all IDEA categories in view: 3 validation and 8 test candidates have a category first seen after `train_end`. Metadata only, no labels | DISCLOSE |
| 17 | RSS articles outside training and evaluation | Only `api/service.py` reads `data/live/`; build and experiment never do | OK |

## Selection and test

| # | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| 18 | Epochs, model, lambda and targets chosen on validation only | Early stopping on validation loss; targets = validation candidate shares | OK |
| 19 | Lambda chosen by a written rule | Rule of `docs/test_protocol.md` section 6, applied on the regenerated validation run: lambda 0.1 (decided 22 Nov) | OK |
| 20 | Test computes only the frozen lambda and the lambda = 0 ablation | `experiment.py`: lambdas on test = {0, frozen lambda}; `--lambdas` refused on test (`tests/test_freeze_protocol.py`) | OK |
| 21 | Every input that changes test numbers is frozen | `python -m agora.freeze create/verify` hashes models, embeddings and encoder, processed data and labels, both configs, source code, test rows and package versions; the test run refuses any difference (`tests/test_freeze_protocol.py`, `tests/test_freeze.py`) | OK |
| 22 | Test run once and logged | Every run is logged in `artifacts/freeze/test_runs.jsonl`; a second run needs `--rerun-reason` and writes `experiment_test_run2`, never over the first | OK |
| 23 | Exclusion counts reported for every ranking | `total_feeds`, `evaluated_feeds` and `excluded_zero_positive_feeds` for every ranking in the JSON, and in the header of the Markdown report | OK |
| 24 | Test labels not inspected beyond preparation counts | Only the pipeline count (1,144 of 2,656 test feeds without a positive) was shared | OK |

## Filled at the freeze

| Item | Value |
| --- | --- |
| Frozen git commit | |
| Freeze manifest SHA-256 | |
| IDEA commit | |
| Checked by / date | |
| Data owner sign-off (Brett) / date | |
