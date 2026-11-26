"""Score articles that are not in the training catalogue (today's news).

Every signal is computed as for IDEA candidates: content similarity to the reader's
profile, topic affinity, freshness, outlet affinity, topic and outlet popularity,
all from the reader's training-period history (profiles frozen at the end of
training). Article-level popularity is unknown for new articles and is not used.

In the live demo, articles the reader clicked today can be added to that history
(``extra``): they count in the content profile, topic and outlet affinity, each one
as TODAY_WEIGHT training reads, because a recent read says more about what the reader
wants now. The models are not retrained.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .neural import _group_popularity


def reader_history(builder, user_id, cutoff):
    """Catalogue rows of the reader's articles read before cutoff (oldest first)."""
    if user_id not in builder.eligible:
        raise ValueError(f'User lacks required distinct training reads: {user_id}')
    reads = builder.reads
    past = reads[(reads.user_id == user_id) & (reads.ts < cutoff)]
    if past.empty:
        raise ValueError(f'Empty eligible history for {user_id}')
    return past


TODAY_WEIGHT = 3


def reader_profile(builder, user_id, cutoff=None, extra_vectors=None):
    """Unit content profile: mean text vector of the training reads plus today's reads."""
    cutoff = cutoff or builder.protocol.train_end
    past = reader_history(builder, user_id, cutoff)
    profile = builder.vectors[[builder.positions[a] for a in past.article_id]].sum(axis=0)
    weight = len(past)
    if extra_vectors is not None and len(extra_vectors):
        profile = profile + TODAY_WEIGHT * np.asarray(extra_vectors, dtype=float).sum(axis=0)
        weight += TODAY_WEIGHT * len(extra_vectors)
    profile = profile / weight
    return profile / max(np.linalg.norm(profile), 1e-12)


def live_features(builder, user_id, articles: pd.DataFrame, vectors, now=None, cutoff=None,
                  extra=None):
    """Return (features in the neural model's order, history rows) for new articles.

    ``articles`` needs topic, outlet and published_at; ``vectors`` are their text
    vectors in the catalogue's space (models.embeddings.load_text_encoder).
    ``extra`` = (vectors, topics, outlets) of articles read today, added to the history.
    """
    cutoff = cutoff or builder.protocol.train_end
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    past = reader_history(builder, user_id, cutoff)
    rows = [builder.positions[a] for a in past.article_id]
    extra_vectors, extra_topics, extra_outlets = extra if extra else (None, [], [])
    profile = reader_profile(builder, user_id, cutoff, extra_vectors)
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    content = vectors @ profile
    meta = builder.articles.loc[past.article_id]
    n = len(past) + TODAY_WEIGHT * len(extra_topics)
    topic_counts = meta.topic.value_counts().add(
        TODAY_WEIGHT * pd.Series(list(extra_topics), dtype=object).value_counts(), fill_value=0)
    outlet_counts = (meta.outlet.value_counts() if 'outlet' in meta else pd.Series(dtype=float)).add(
        TODAY_WEIGHT * pd.Series(list(extra_outlets), dtype=object).value_counts(), fill_value=0)
    topic_pop, outlet_pop = _group_popularity(builder, cutoff)
    published = pd.to_datetime(articles.published_at, utc=True, errors='coerce')
    hours = ((now - published).dt.total_seconds() / 3600).fillna(24 * 7).clip(lower=0)
    features = np.column_stack([
        content,
        [topic_counts.get(t, 0) / n for t in articles.topic],
        (1.0 / (1.0 + hours / 24.0)).to_numpy(),
        [outlet_counts.get(o, 0) / n for o in articles.outlet],
        [topic_pop.get(t, 0.0) for t in articles.topic],
        [outlet_pop.get(o, 0.0) for o in articles.outlet],
    ])
    return features, rows
