"""Point-in-time profiles over canonical first-read events."""
from dataclasses import dataclass
import pandas as pd
import numpy as np


def aware_timestamp(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"Timezone-aware timestamp required: {value!r}")
    return timestamp.tz_convert('UTC')


@dataclass(frozen=True)
class TemporalProtocol:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp
    min_train_articles: int = 5

    @classmethod
    def from_config(cls, config):
        values = [aware_timestamp(config[key]) for key in
                  ('train_start', 'train_end', 'validation_end', 'test_end')]
        if not all(left < right for left, right in zip(values, values[1:])):
            raise ValueError('Split boundaries must be strictly increasing')
        minimum = config.get('min_train_articles', 5)
        if not isinstance(minimum, int) or minimum < 1:
            raise ValueError('min_train_articles must be a positive integer')
        return cls(*values, minimum)

    def cutoff(self, generated_at, split):
        timestamp = aware_timestamp(generated_at)
        bounds = {'train': (self.train_start, self.train_end),
                  'validation': (self.train_end, self.validation_end),
                  'test': (self.validation_end, self.test_end)}
        if split not in bounds:
            raise ValueError(f'Unknown split: {split}')
        start, end = bounds[split]
        if not start <= timestamp < end:
            raise ValueError(f'Feed outside declared {split} interval: {timestamp}')
        return timestamp if split == 'train' else self.train_end


class ProfileBuilder:
    def __init__(self, reads, articles, embeddings, protocol):
        self.protocol = protocol
        self.articles = articles.set_index('article_id', verify_integrity=True)
        ids = articles.article_id.tolist()
        vectors = np.asarray(embeddings, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids) or not np.isfinite(vectors).all():
            raise ValueError('Invalid article embedding matrix')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)
        self.positions = {article: index for index, article in enumerate(ids)}
        self.reads = reads.copy()
        self.reads['ts'] = pd.to_datetime([aware_timestamp(t) for t in self.reads.ts], utc=True)
        if ((self.reads.ts < protocol.train_start) | (self.reads.ts >= protocol.train_end)).any():
            raise ValueError('reads_train contains events outside the training interval')
        unknown = set(self.reads.article_id) - set(ids)
        if unknown:
            raise ValueError(f'Unknown history article IDs: {sorted(unknown)[:5]}')
        self.reads = self.reads.sort_values('ts').drop_duplicates(['user_id', 'article_id'])
        counts = self.reads.groupby('user_id').article_id.nunique()
        self.eligible = set(counts[counts >= protocol.min_train_articles].index)

    def features(self, user_id, candidates, cutoff):
        if user_id not in self.eligible:
            raise ValueError(f'User lacks required distinct training reads: {user_id}')
        # Strict inequality excludes the target event and simultaneous events.
        past = self.reads[self.reads.ts < cutoff]
        history = past[past.user_id == user_id].article_id.tolist()
        if not history:
            raise ValueError(f'Empty eligible history for {user_id} at {cutoff}')
        missing = set(candidates) - set(self.positions)
        if missing:
            raise ValueError(f'Unknown candidates: {sorted(missing)[:5]}')
        profile = self.vectors[[self.positions[a] for a in history]].mean(axis=0)
        profile /= max(np.linalg.norm(profile), 1e-12)
        topic_counts = self.articles.loc[history, 'topic'].value_counts()
        readers = past.groupby('article_id').user_id.nunique()
        reader_count = max(past.user_id.nunique(), 1)
        candidate_vectors = self.vectors[[self.positions[a] for a in candidates]]
        similarity = candidate_vectors @ profile
        affinity = [topic_counts.get(self.articles.loc[a, 'topic'], 0) / len(history)
                    for a in candidates]
        popularity = [readers.get(a, 0) / reader_count for a in candidates]
        return np.column_stack([similarity, affinity, popularity]), len(history)
