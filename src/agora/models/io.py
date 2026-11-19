"""Strict inputs, auditable hashes and UTF-8 JSON/JSONL support."""
import hashlib
import json
from pathlib import Path
import pandas as pd


def read_records(path, wrapper=None):
    path = Path(path)
    if path.suffix == '.csv':
        records = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records')
    elif path.suffix == '.parquet':
        records = pd.read_parquet(path).to_dict('records')
    else:
        text = path.read_text(encoding='utf-8-sig')
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(records, dict) and wrapper:
            records = records.get(wrapper)
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        raise ValueError(f'Expected nonempty records in {path}')
    return records


def require_ids(frame, columns):
    for column in columns:
        if column not in frame or frame[column].isna().any():
            raise ValueError(f'Missing {column}')
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'{column} must contain nonempty string IDs')
        frame[column] = frame[column].str.strip()
    return frame


def load_articles(path):
    frame = require_ids(pd.DataFrame(read_records(path, 'articles')), ['article_id'])
    for column in ('title', 'lead', 'topic'):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f'Article field missing: {column}')
        frame[column] = frame[column].astype(str).str.strip()
    if frame.article_id.duplicated().any():
        raise ValueError('Duplicate canonical article IDs')
    if (frame.title.eq('') & frame.lead.eq('')).any() or frame.topic.eq('').any():
        raise ValueError('Each article requires text and an explicit topic')
    return frame.sort_values('article_id').reset_index(drop=True)


def load_reads(path):
    frame = require_ids(pd.DataFrame(read_records(path, 'reads')), ['user_id', 'article_id'])
    if 'ts' not in frame:
        raise ValueError('reads_train requires ts timestamps')
    return frame[['user_id', 'article_id', 'ts']]


def load_feeds(path):
    feeds = read_records(path, 'feeds')
    ids = set()
    for feed in feeds:
        for key in ('user_id', 'feed_id'):
            value = feed.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f'Feed {key} must be a canonical string ID')
        if feed['feed_id'] in ids:
            raise ValueError('Duplicate feed_id')
        ids.add(feed['feed_id'])
        candidates = feed.get('candidates')
        if not isinstance(candidates, list) or not candidates:
            raise ValueError('Expected a nonempty candidates list')
        if not all(isinstance(a, str) and a and a == a.strip() for a in candidates):
            raise ValueError('Candidate IDs must be canonical nonempty strings')
        if len(set(candidates)) != len(candidates):
            raise ValueError('Duplicate candidate IDs')
        if 'generated_at' not in feed:
            raise ValueError('Feed timestamp missing')
    return feeds


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
