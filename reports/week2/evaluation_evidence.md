# Evaluation V1 evidence — 15 November 2026
Owner: Jerry Cheng. Branch: jerry/evaluation-v1.

## Reproduction
From the repository root, after installing the package:
```powershell
python -m agora.evaluation --predictions examples/mock_predictions.json --labels examples/mock_labels.json --output artifacts/evaluation/mock_metrics.json
python -m unittest discover -s tests -v
```
Alternatively set `$env:PYTHONPATH = "$PWD/src"` before running, without package installation.

## Observed results
- 13 unittest tests passed.
- 3 synthetic feeds of 26 candidates: 2 evaluated, 1 excluded for no positives.
- Macro NDCG@10: 0.7346393630113782.
- Macro Recall@20: 0.8333333333333333.
- Full machine-readable output: evaluation_mock_results.json.
- Binary NDCG agrees with the hand calculation documented in the evaluation README.

## Week 2 Q3
Yes: the evaluation harness runs on mock predictions and returns ranking metrics.
This is evidence of evaluator functionality, not real recommendation performance.
Add the commit/PR link after publishing.

## Pending
The team has agreed the chronological split and first-read window; see docs/temporal_protocol.md. Verify source timestamps, canonical IDs and
final-feed policy with Brett and Ronald. Integrate their actual predictions and
separately generated held-out labels. Other Week 2 questions remain unassessed.