# api
Owner: Mathis.

FastAPI service: `python -m uvicorn agora.api.main:app` (endpoints in the root README).

- Real mode: at start-up, `service.py` loads the processed feeds (validation and test), the article catalogue, the training reads, the embeddings and text encoder, the neural model (`artifacts/models/neural.pt`; the logistic `model.json` if no neural model exists) and `configs/reranking_agents.json`. `/recommend` scores the feed's candidates with profiles frozen at the end of training, reranks them with SCRUF-D and returns every candidate with its rank, metadata, `relevance_score`, `reranking_score` and `latency_ms`. Labels are never loaded.
- Mock mode: when those files are missing or `AGORA_MOCK=1`, the synthetic feed of `examples/feed_input.json` is returned (`is_mock: true`).
- Live mode: `/recommend_live?user_id=...` works in two stages on today's articles (`data/live/articles_live.json`, from `python -m agora.data.live`). First, candidate generation builds a personal pool of 26 articles: the reader's closest article in every topic, then the closest overall. Second, the same model and profile score the pool, then SCRUF-D reranks it (own exposure state). The file is reloaded when it changes. `/live/status` says whether articles are loaded.
- `POST /log_interaction` validates `display` / `click` events and stores them in SQLite (`src/agora/data/storage.py`, file `artifacts/agora.db`, override with `AGORA_DB`).
- Paths can be overridden with `AGORA_DATA_DIR`, `AGORA_MODEL_DIR`, `AGORA_CONFIG`, `AGORA_RERANK_CONFIG`.

Measured on IDEA (CPU, 19 Nov 2026): start-up about 5 s; `/recommend` about 60 ms per feed of 26 candidates with the neural model; `/recommend_live` about 25 ms.
