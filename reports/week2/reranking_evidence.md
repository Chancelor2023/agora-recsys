# Reranking V1 evidence — 15 November 2026
Owner: Romain Cortyl. Branch: romain/reranking-v1.

## Reproduction
From the repository root, after installing the package (Python 3.11+):
```powershell
python -m agora.reranking --input examples/reranking_mock_input.json --config configs/reranking_agents.json --output artifacts/reranking/mock_predictions.json
python -m agora.reranking --input examples/reranking_mock_input.json --lambda 0 --output artifacts/reranking/mock_predictions_ablation.json
python -m agora.evaluation --predictions artifacts/reranking/mock_predictions.json --labels examples/reranking_mock_labels.json
python -m unittest discover -s tests -v
```

## Observed results
- Full forward pass (allocation, rescoring choice, exposure-state update) on 3 synthetic feeds of 12 candidates: 36 ranked rows, contiguous ranks, no error; state updated 3 times.
- Selected agents: feed 1 none (no history yet), feed 2 `environment`, feed 3 `public_affairs` (lottery, seed 42).
- Feed 3 top-10 position-weighted exposure of the `public_affairs` group: 0.409 with SCRUF-D vs 0.234 for the relevance-only ablation (`--lambda 0`).
- The output is accepted by the evaluation harness: 3/3 feeds evaluated. Synthetic metrics (interface check only, arbitrary labels): NDCG@10 0.837 (SCRUF-D) vs 0.818 (ablation); Recall@20 1.0 for both.
- 12 reranking unittest tests pass (30 with the evaluation tests).

## Week 2 Q2
Yes: the SCRUF-D-inspired reranker completes a forward pass and updates its exposure state between feeds.
This is evidence of functionality on fictional scores, not of recommendation quality or fairness on IDEA.
Add the commit/PR link after publishing.

## Pending
Confirm IDEA topic values and declared targets (`configs/reranking_agents.json` is provisional). Plug in Ronald's relevance scores, then measure exposure fairness and accuracy against the ablation on validation data.
