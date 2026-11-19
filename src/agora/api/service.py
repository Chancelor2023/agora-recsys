"""Real recommendation service: relevance model + SCRUF-D reranker on IDEA feeds.

Loaded once at API start-up. It serves the validation and test feeds exported by
the data pipeline, with reader profiles frozen at the end of training. Labels are
never loaded. The reranker keeps one in-memory exposure state for the live API.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERVED_SPLITS = ("validation", "test")


@dataclass
class ServicePaths:
    data_dir: Path
    model_dir: Path
    config: Path
    rerank_config: Path
    live_articles: Path = field(default_factory=lambda: Path("data/live/articles_live.json"))

    @classmethod
    def from_env(cls, root: Path) -> "ServicePaths":
        def path(name: str, default: str) -> Path:
            value = Path(os.environ.get(name, default))
            return value if value.is_absolute() else root / value
        return cls(path("AGORA_DATA_DIR", "data/processed"),
                   path("AGORA_MODEL_DIR", "artifacts/models"),
                   path("AGORA_CONFIG", "configs/experiment.toml"),
                   path("AGORA_RERANK_CONFIG", "configs/reranking_agents.json"),
                   path("AGORA_LIVE_ARTICLES", "data/live/articles_live.json"))

    def ready(self) -> bool:
        model_files = [self.model_dir / "neural.pt", self.model_dir / "model.json"]
        if not any(p.exists() for p in model_files):
            return False
        needed = [self.data_dir / "articles.json", self.data_dir / "reads_train.csv",
                  self.model_dir / "embeddings.npz",
                  self.config, self.rerank_config]
        needed += [self.data_dir / f"feeds_{s}.jsonl" for s in SERVED_SPLITS]
        return all(p.exists() for p in needed)


class RecommenderService:
    def __init__(self, predictor, articles: pd.DataFrame, feeds: dict[str, dict[str, Any]],
                 reranker, model_version: str, live_reranker=None, text_encoder=None,
                 live_path: Path | None = None) -> None:
        self.predictor = predictor
        self.live_reranker = live_reranker
        self.text_encoder = text_encoder
        self.live_path = live_path
        self.live_pool = 26  # size of the personal candidate pool, as an IDEA daily feed
        self._live = None  # (mtime, articles DataFrame, vectors)
        self.articles = articles.set_index("article_id", drop=False)
        self.feeds = feeds
        self.reranker = reranker
        self.model_version = model_version
        self._by_user: dict[str, list[dict[str, Any]]] = {}
        for feed in sorted(feeds.values(), key=lambda f: f["generated_at"]):
            self._by_user.setdefault(feed["user_id"], []).append(
                {"feed_id": feed["feed_id"], "generated_at": feed["generated_at"],
                 "split": feed["split"]})

    @classmethod
    def from_paths(cls, paths: ServicePaths) -> "RecommenderService":
        from agora.models.config import load_model_config
        from agora.models.embeddings import load_embeddings
        from agora.models.io import load_articles, load_feeds, load_reads
        from agora.models.embeddings import load_text_encoder
        from agora.models.relevance import FeedPredictor, RelevanceModel
        from agora.profiles import ProfileBuilder, TemporalProtocol
        from agora.reranking.io import build_reranker

        config = load_model_config(paths.config)
        articles_full = pd.DataFrame(json.loads((paths.data_dir / "articles.json")
                                                .read_text(encoding="utf-8")))
        articles = load_articles(paths.data_dir / "articles.json")
        vectors, _ = load_embeddings(paths.model_dir / "embeddings.npz", articles)
        protocol = TemporalProtocol.from_config(config)
        builder = ProfileBuilder(load_reads(paths.data_dir / "reads_train.csv"),
                                 articles, vectors, protocol)
        if (paths.model_dir / "neural.pt").exists():
            from agora.models.neural import NeuralFeedPredictor, NeuralRecommender
            neural, payload = NeuralRecommender.load(paths.model_dir / "neural.pt", builder)
            predictor = NeuralFeedPredictor(builder, neural)
            model_name = payload["model_version"]
        else:
            model, payload = RelevanceModel.load(paths.model_dir / "model.json")
            predictor = FeedPredictor(builder, model)
            model_name = f"logistic-{payload['model_version']}"
        feeds: dict[str, dict[str, Any]] = {}
        for split in SERVED_SPLITS:
            for feed in load_feeds(paths.data_dir / f"feeds_{split}.jsonl"):
                feeds[feed["feed_id"]] = {**feed, "split": split}
        rerank_config = json.loads(paths.rerank_config.read_text(encoding="utf-8"))
        reranker = build_reranker(rerank_config)
        live_reranker = build_reranker(rerank_config)  # its own exposure state
        try:
            encoder = load_text_encoder(paths.model_dir / "embeddings.npz")
        except (OSError, ValueError, ImportError):
            encoder = None  # live recommendations unavailable until the encoder is rebuilt
        version = f"{model_name}+scruf-d-lambda-{reranker.lam}"
        return cls(predictor, articles_full, feeds, reranker, version, live_reranker,
                   encoder, paths.live_articles)

    # ------------------------------------------------------------------ #
    def users(self) -> list[str]:
        return sorted(self._by_user)

    def feeds_for(self, user_id: str) -> list[dict[str, Any]]:
        if user_id not in self._by_user:
            raise KeyError(f"Unknown user_id: {user_id}")
        return self._by_user[user_id]

    def recommend(self, user_id: str, feed_id: str) -> list[dict[str, Any]]:
        feed = self.feeds.get(feed_id)
        if feed is None:
            raise KeyError(f"Unknown feed_id: {feed_id}")
        if feed["user_id"] != user_id:
            raise ValueError(f"Feed {feed_id} does not belong to user {user_id}")
        rows = self.predictor.predict_feed(user_id, feed_id, feed["generated_at"],
                                           feed["candidates"], feed["split"])
        candidates = []
        for row in rows:
            meta = self.articles.loc[row["article_id"]]
            candidates.append({"article_id": row["article_id"],
                               "relevance_score": row["relevance_score"],
                               "topic": meta["topic"], "outlet": meta["outlet"]})
        result = self.reranker.rerank({"user_id": user_id, "feed_id": feed_id,
                                       "candidates": candidates})
        items = []
        for row in result.ranking:
            meta = self.articles.loc[row["article_id"]]
            items.append({
                "article_id": row["article_id"], "rank": row["rank"],
                "title": meta["title"], "lead": meta["lead"], "outlet": meta["outlet"],
                "published_at": meta["published_at"], "topic": meta["topic"],
                "url": meta.get("url") or None, "image": meta.get("image") or None,
                "relevance_score": row["relevance_score"],
                "reranking_score": row["reranking_score"],
            })
        return items

    # ------------------------------------------------------------------ #
    # Today's news (RSS), scored with the same model and profiles
    # ------------------------------------------------------------------ #
    def live_status(self) -> dict[str, Any]:
        if self.text_encoder is None:
            return {"available": False, "reason": "text encoder missing: rerun python -m agora.build"}
        if self.live_path is None or not self.live_path.exists():
            return {"available": False, "reason": "no live articles: run python -m agora.data.live"}
        live = self._load_live()
        return {"available": True, "articles": len(live[1]),
                "fetched_at": live[1].fetched_at.max() if "fetched_at" in live[1] else None,
                "topics": live[1].topic.value_counts().to_dict()}

    def _load_live(self):
        mtime = self.live_path.stat().st_mtime
        if self._live is None or self._live[0] != mtime:
            frame = pd.DataFrame(json.loads(self.live_path.read_text(encoding="utf-8")))
            frame = frame.drop_duplicates("article_id").reset_index(drop=True)
            vectors = self.text_encoder((frame.title + "\n" + frame.lead).tolist())
            self._live = (mtime, frame, vectors)
        return self._live

    def recommend_live(self, user_id: str, limit: int = 20, now=None) -> list[dict[str, Any]]:
        from agora.models.live import live_features

        status = self.live_status()
        if not status["available"]:
            raise LookupError(status["reason"])
        _, all_articles, all_vectors = self._load_live()
        builder = self.predictor.builder
        # Stage 1, candidate generation: a personal pool of `pool` articles, the reader's
        # closest article in every topic (coverage) then the closest overall (interest).
        # Stage 2 ranks this pool, the setting the model was trained in (IDEA feeds of ~26).
        keep = _personal_pool(builder, user_id, all_articles, all_vectors, self.live_pool)
        frame = all_articles.iloc[keep].reset_index(drop=True)
        vectors = all_vectors[keep]
        features, history = live_features(builder, user_id, frame, vectors, now=now)
        model = self.predictor.model
        if hasattr(model, "score_articles"):
            scores = model.score_articles(history, vectors, frame.topic.tolist(),
                                          frame.outlet.tolist(), features)
        else:  # logistic model: content similarity, topic affinity, popularity (unknown: 0)
            scores = model.predict(np.column_stack([features[:, 0], features[:, 1],
                                                    np.zeros(len(frame))]))
        candidates = [{"article_id": a, "relevance_score": float(s), "topic": t, "outlet": o}
                      for a, s, t, o in zip(frame.article_id, scores, frame.topic, frame.outlet)]
        result = self.live_reranker.rerank({"user_id": user_id, "feed_id": "live", "candidates": candidates})
        by_id = frame.set_index("article_id")
        items = []
        for row in result.ranking[:limit]:
            meta = by_id.loc[row["article_id"]]
            items.append({"article_id": row["article_id"], "rank": row["rank"], "title": meta["title"],
                          "lead": meta["lead"], "outlet": meta["outlet"],
                          "published_at": meta["published_at"], "topic": meta["topic"],
                          "url": meta.get("url") or None, "image": meta.get("image") or None,
                          "relevance_score": row["relevance_score"],
                          "reranking_score": row["reranking_score"]})
        return items


def _personal_pool(builder, user_id, articles, vectors, size):
    """Indices of a personal candidate pool: best per topic, then best overall."""
    from agora.models.live import reader_history

    past = reader_history(builder, user_id, builder.protocol.train_end)
    profile = builder.vectors[[builder.positions[a] for a in past.article_id]].mean(axis=0)
    profile /= max(np.linalg.norm(profile), 1e-12)
    norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
    similarity = (vectors @ profile) / norms
    order = list(np.argsort(-similarity, kind="stable"))
    chosen, seen_topics = [], set()
    for i in order:  # coverage: the closest article of each topic
        topic = articles.topic.iloc[i]
        if topic not in seen_topics:
            seen_topics.add(topic)
            chosen.append(i)
    for i in order:  # interest: fill with the closest articles overall
        if len(chosen) >= size:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen[:size])
