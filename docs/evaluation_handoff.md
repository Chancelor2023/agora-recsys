# Real-data evaluation handoff
The metric harness is ready; real data and predictions have not been supplied.

## Brett
Place local exports under data/processed/:
- validation_labels.json and test_labels.json:
  a list or {"labels": [...]} with feed_id, article_id, label (integer 0 or 1).
- Feed metadata and eligible histories for Ronald, kept separate from labels.
- An audit recording timezone, timestamp semantics, observation end, canonical
  ID mapping, source revision, split counts, candidate counts, five-training-read
  eligibility counts, shortened windows and unresolved joins.
Apply docs/temporal_protocol.md. Confirm canonicalisation before joining.

## Ronald
Provide validation_predictions.json and test_predictions.json under data/processed/.
Use a list or {"predictions": [...]}:
```json
{
  "user_id": "reader-id",
  "feed_id": "feed-id",
  "article_id": "canonical-article-id",
  "rank": 1,
  "relevance_score": 0.83
}
```
Scores need not be probabilities. Each feed needs every candidate exactly once
and ranks 1..n. Define deterministic score tie-breaking. The evaluator uses
explicit rank, allowing scores to be retained unchanged after reranking.

## Jerry
After receiving and checking the audit, evaluate validation:
```powershell
python -m agora.evaluation --config configs/experiment.toml --predictions data/processed/validation_predictions.json --labels data/processed/validation_labels.json --output artifacts/evaluation/validation_metrics.json
```
Test is evaluated only by the declared run of `docs/test_protocol.md`
(`python -m agora.freeze create`, then `python -m agora.experiment --split test`);
`agora.models predict --split test` is refused.
Use --scope supplementary with separately filtered matching inputs for @5 metrics.
Keep real participant-level inputs and per-feed outputs outside version control.
Only publish approved aggregate summaries.

## Readiness checklist
- [ ] Source timezone and timestamp meanings verified.
- [ ] Canonical IDs and first-read/window rules implemented and checked.
- [ ] Training-feed histories contain only earlier reads.
- [ ] Validation/test profiles and learned quantities use training reads only.
- [ ] Cutoffs selected on validation; test untouched until frozen.
- [ ] Feed and candidate sets match, with audit counts.
- [ ] Actual evaluator command and output archived.
