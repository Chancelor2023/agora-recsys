# Proposed component interfaces
This is a starting contract for team agreement, not an implemented API.

## Identifiers and timestamps
Treat IDs as strings. A feed identifies one reader's candidate list at a given time. Retain a mapping between original article IDs, variants and canonical content IDs. Agree on the canonicalisation policy before joining records. Use timezone-aware timestamps after validating the source timezone.

## Shared example
`examples/feed_input.json` is entirely synthetic. Its two candidates demonstrate the schema only; they are not sufficient for a real @10/@20 experiment.
Labels live separately in `examples/evaluation_labels.json`.

## Data to relevance scoring
Input: user_id, feed_id, generated_at, eligible history and candidate metadata.
Output: one relevance_score per candidate, retaining user_id, feed_id and article_id.
Use only eligible past reads; never provide the evaluation label to the scoring function.

## Relevance scoring to reranking
Input: candidate scores, topic metadata, declared targets and past recommendation exposure state.
Output: unique article IDs with contiguous ranks starting at 1, selected agent and updated state.
Preserve relevance scores separately from reranking scores. Evaluate each algorithm with its own exposure state.

## Predictions to evaluation
Predictions: user_id, feed_id, article_id, rank, relevance_score.
Separate labels: feed_id, article_id, label (0 or 1).
Join within feeds and evaluate identical candidate sets. Report feeds excluded for missing positives. A zero label means no observed first read in the declared window, not proven dislike.

## API and interaction logging
Proposed GET /recommend?user_id=...&feed_id=...
Response: user_id, feed_id, model_version, is_mock, items ordered by rank.
Each item includes article_id, rank, title, lead, outlet and published_at.
Proposed POST /log_interaction: event_id, user_id, feed_id, article_id, event_type, timestamp, displayed_rank and model_version.
Use event_type display or click. The backend should validate and persist actual application events; source recommendation records remain distinguishable.

## Validated temporal agreement
See [temporal_protocol.md](temporal_protocol.md) for the agreed methodology and [evaluation_handoff.md](evaluation_handoff.md) for the real-data handoff. Agreement does not replace verification of source timestamps and canonical IDs.
