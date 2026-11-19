# reranking
Owner: Romain.

SCRUF-D-inspired agents, allocation, rescoring and exposure state (Aird et al.).

## Forward pass for one feed
1. **Allocation** (`ScrufDReranker.allocate`): each fairness agent protects a topic group. Its fairness is the observed position-weighted top-k exposure share of that group over the last `window` delivered lists, divided by its declared `target_share` (capped at 1). Weight = `(1 - fairness) * compatibility`. `lottery` (default) draws one agent proportionally to these weights; `weighted` keeps all agents.
2. **Choice** (`ScrufDReranker.choose`): min-max normalised `relevance_score` + `lambda * sum(weight_i * 1[article in group_i])`. Ties: relevance, then article ID.
3. **State update** (`ExposureState.update`): stores each group's exposure in the delivered top-k list (weight `1/log2(rank+1)`).

`--lambda 0` is the ablation (relevance ranking only). Each algorithm keeps its own exposure state. Feeds are processed in `generated_at` order.

## Contract
Input: `{"feeds": [{"user_id", "feed_id", "generated_at", "candidates": [{"article_id", "relevance_score", "topic", ...}]}]}` or the relevance model's flat `predictions` rows carrying `topic`. Labels are rejected.
Output: `predictions` rows (`user_id`, `feed_id`, `article_id`, contiguous `rank` from 1, unchanged `relevance_score`, separate `reranking_score`), per-feed `selections` (selected agent, weights, fairness) and `final_state`.
Agents and targets are declared in `configs/reranking_agents.json` (provisional until IDEA topics are confirmed).

## Run
```
python -m agora.reranking --input examples/reranking_mock_input.json --config configs/reranking_agents.json --output artifacts/reranking/mock_predictions.json
python -m agora.evaluation --predictions artifacts/reranking/mock_predictions.json --labels examples/reranking_mock_labels.json
python -m unittest tests.test_reranking -v
```
The mock input uses fictional scores and topics; its metrics are interface checks, not results.
