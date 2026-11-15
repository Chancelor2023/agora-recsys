# Evaluation V1 — Jerry
Standard-library-only binary relevance evaluation. No trained model is needed.

## Commands
After installing the project with `python -m pip install -r requirements.txt`:
```powershell
python -m agora.evaluation --predictions examples/mock_predictions.json --labels examples/mock_labels.json --output artifacts/evaluation/mock_metrics.json
python -m unittest discover -s tests -v
```
Alternatively, without installation, set `$env:PYTHONPATH = "$PWD/src"` first.

## Input contract
JSON lists or wrappers with a predictions/labels list are accepted.
Prediction rows require user_id, feed_id, article_id, rank and finite relevance_score.
Label rows require feed_id, article_id and binary integer label.
Each feed must include the complete candidate list, with identical candidate IDs
in both inputs, one reader and unique contiguous ranks starting at 1.
Truncated top-k predictions, missing labels, duplicate IDs/ranks and mismatched
feeds are rejected, rather than silently changing the evaluation universe.
Scores are retained for traceability; explicit rank controls the result.
Resolve score ties upstream using a documented deterministic rule.

## Metrics
For ordered binary labels y_i, DCG@k = sum(y_i / log2(i+1)), i=1..min(k,n).
IDCG places all available positives first, up to k. NDCG = DCG / IDCG.
Recall@k = positives in first min(k,n) ranks / positives in the entire feed.
Both metrics range from 0 to 1 for feeds with positives.
See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
for the NDCG definition. This implementation uses explicit ranks and binary labels.

## Aggregation and edge cases
Compute each metric per feed, then average equally across eligible feeds.
This is not an equal-weight average across readers; frequent readers can have
more feeds. Feeds without positives are excluded from both averages and counted.
Their per-feed values are null. If all feeds are excluded, macro values are null.
For fewer than k candidates, use all candidates without padding or changing
the recall denominator. Use --ndcg-k 5 --recall-k 5 for supplementary analysis.

## Hand-checkable example
The three fictional feeds each contain 26 candidates.
- mock-a: positives at ranks 1, 11, 21. NDCG@10 = 1 / (1 + 1/log2(3) + 1/2);
  Recall@20 = 2/3.
- mock-b: positives at ranks 1, 2. Both metrics = 1.
- mock-empty: no positives, excluded and reported.

Expected macro NDCG@10 = approximately 0.734639; Recall@20 = 0.833333.
These numbers validate code, not Agora quality. The older two-candidate
integration example remains unchanged and can also be evaluated after creating
matching prediction rows.

## Remaining work
Agree on canonical IDs, temporal label attribution and exclusion counts with
Brett and Ronald; then use actual held-out feed predictions. Chronology and
label construction are upstream responsibilities and are not verified merely
by computing these metrics. Diversity, coverage and ablation metrics remain to add.

## Shared configuration and real-data integration
Pass --config configs/experiment.toml to read evaluation cutoffs. Use --scope supplementary for @5; explicit --ndcg-k and --recall-k override configured values. The output records effective cutoffs. This loads metric settings only and does not enforce temporal preprocessing. See docs/temporal_protocol.md and docs/evaluation_handoff.md.
