"""Neural news recommender inspired by NRMS (Wu et al., EMNLP 2019).

Architecture
- News encoder: article text vector (TF-IDF+SVD or sentence embedding) projected
  to a hidden space, plus learned topic and outlet embeddings.
- User encoder: multi-head self-attention over the reader's reading history
  (articles read strictly before the feed, at most ``max_history``), followed by
  additive attention pooling into one user vector (as in NRMS).
- Click predictor (Wide & Deep): a linear "wide" part over five point-in-time
  signals (content similarity, topic affinity, training popularity, freshness,
  outlet affinity) plus a "deep" MLP over (user vector * candidate vector,
  user-candidate cosine, signals). The deep part starts at zero, so training
  begins at the linear solution and only adds what improves validation loss.

Training is pointwise binary cross-entropy over every candidate of a feed,
epoch by epoch with AdamW, with early stopping on the validation loss. Profiles
follow the temporal protocol: training feeds use reads strictly before the
feed; validation and test use profiles frozen at the end of training. Test
labels are never read.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .relevance import rank_candidates


@dataclass
class NeuralConfig:
    hidden: int = 64
    heads: int = 4
    max_history: int = 50
    dropout: float = 0.3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    batch_feeds: int = 64
    epochs: int = 60
    patience: int = 8
    seed: int = 42
    loss: str = 'listwise'  # 'listwise' (softmax over the feed, as NRMS) or 'pointwise' (BCE)
    position_debias: bool = True  # PAL: learn the display-position effect in training only
    personal_signals_only: bool = True  # drop freshness / topic / outlet popularity (same for every reader)


# --------------------------------------------------------------------------- #
# Examples: one per feed (history + all candidates)
# --------------------------------------------------------------------------- #
MAX_POSITION = 32  # served positions beyond this share one bias
N_FEATURES = 6  # content similarity, topic affinity, freshness, outlet affinity, topic popularity, outlet popularity


def _group_popularity(builder, cutoff):
    """Share of readers' reads per topic and per outlet, from reads before cutoff."""
    key = ('group_pop', cutoff)
    cache = builder.__dict__.setdefault('_agora_cache', {})
    if key not in cache:
        before = builder.reads[builder.reads.ts < cutoff]
        meta = builder.articles.loc[before.article_id]
        n = max(len(before), 1)
        topic = (meta.topic.value_counts() / n).to_dict()
        outlet = (meta.outlet.value_counts() / n).to_dict() if 'outlet' in meta else {}
        if len(cache) > 512:
            cache.clear()
        cache[key] = (topic, outlet)
    return cache[key]


def _extra_features(builder, feed, past, cutoff):
    """Freshness, outlet affinity, topic and outlet popularity: all known when the feed is served.

    Article-level popularity is useless for new articles (none of the validation
    candidates was read during training), so popularity is taken at topic and
    outlet level instead.
    """
    articles = builder.articles
    topic_pop, outlet_pop = _group_popularity(builder, cutoff)
    when = pd.Timestamp(feed['generated_at'])
    when = when.tz_localize('UTC') if when.tzinfo is None else when.tz_convert('UTC')
    fresh, outlet_aff = [], []
    outlets = articles.loc[past.article_id, 'outlet'].value_counts() if 'outlet' in articles else {}
    n = max(len(past), 1)
    for a in feed['candidates']:
        published = articles.at[a, 'published_at'] if 'published_at' in articles else None
        hours = 0.0
        if isinstance(published, str) and published:
            hours = max((when - pd.Timestamp(published)).total_seconds() / 3600, 0.0)
        fresh.append(1.0 / (1.0 + hours / 24.0))  # 1 = just published, 0.5 = one day old
        outlet = articles.at[a, 'outlet'] if 'outlet' in articles else None
        outlet_aff.append(float(outlets.get(outlet, 0)) / n if outlet is not None else 0.0)
    tpop = [topic_pop.get(articles.at[a, 'topic'], 0.0) for a in feed['candidates']]
    opop = [outlet_pop.get(articles.at[a, 'outlet'], 0.0) if 'outlet' in articles else 0.0
            for a in feed['candidates']]
    return np.column_stack([fresh, outlet_aff, tpop, opop])


def build_examples(builder, feeds, split, labels=None, max_history=50, strict=False):
    """Turn feeds into model inputs; feeds without prior history are skipped."""
    target = None
    if labels is not None:
        target = labels.set_index(['feed_id', 'article_id']).label
    reads = builder.reads  # sorted by time, first read per (user, article)
    by_user = {u: g for u, g in reads.groupby('user_id')}
    examples, skipped = [], {'empty_history': 0, 'ineligible_user': 0}
    for feed in feeds:
        user = feed['user_id']
        if user not in builder.eligible:
            if strict:
                raise ValueError(f'Feed contains an ineligible user: {user}')
            skipped['ineligible_user'] += 1
            continue
        cutoff = builder.protocol.cutoff(feed['generated_at'], split)
        past = by_user.get(user)
        past = past[past.ts < cutoff] if past is not None else None
        if past is None or past.empty:
            skipped['empty_history'] += 1
            continue
        base, _ = builder.features(user, feed['candidates'], cutoff)
        # drop article-level popularity (column 2): always 0 for new articles
        features = np.column_stack([base[:, :2], _extra_features(builder, feed, past, cutoff)])
        history = past.article_id.tolist()[-max_history:]
        example = {'feed_id': feed['feed_id'], 'user_id': user,
                   'candidates': list(feed['candidates']),
                   'history': [builder.positions[a] for a in history],
                   'cand_idx': [builder.positions[a] for a in feed['candidates']],
                   'features': features.astype(np.float32)}
        if target is not None:
            example['labels'] = [int(target.loc[(feed['feed_id'], a)]) for a in feed['candidates']]
        examples.append(example)
    return examples, skipped


def _collate(examples, max_history):
    b = len(examples)
    c_max = max(len(e['cand_idx']) for e in examples)
    hist = torch.zeros(b, max_history, dtype=torch.long)
    hist_mask = torch.zeros(b, max_history, dtype=torch.bool)
    cand = torch.zeros(b, c_max, dtype=torch.long)
    cand_mask = torch.zeros(b, c_max, dtype=torch.bool)
    feats = torch.zeros(b, c_max, N_FEATURES)
    labels = torch.zeros(b, c_max)
    positions = torch.arange(c_max).clamp(max=MAX_POSITION - 1).expand(b, c_max).clone()
    for i, e in enumerate(examples):
        h = e['history'][-max_history:]
        hist[i, :len(h)] = torch.tensor(h)
        hist_mask[i, :len(h)] = True
        n = len(e['cand_idx'])
        cand[i, :n] = torch.tensor(e['cand_idx'])
        cand_mask[i, :n] = True
        feats[i, :n] = torch.from_numpy(e['features'])
        if 'labels' in e:
            labels[i, :n] = torch.tensor(e['labels'], dtype=torch.float32)
    return hist, hist_mask, cand, cand_mask, feats, labels, positions


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class AdditiveAttention(nn.Module):
    def __init__(self, dim, query_dim=64):
        super().__init__()
        self.proj = nn.Linear(dim, query_dim)
        self.query = nn.Linear(query_dim, 1, bias=False)

    def forward(self, x, mask):
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class NRMSRecommender(nn.Module):
    def __init__(self, text_vectors, topic_ids, outlet_ids, n_topics, n_outlets, cfg: NeuralConfig):
        super().__init__()
        self.register_buffer('text', torch.as_tensor(text_vectors, dtype=torch.float32))
        self.register_buffer('topic', torch.as_tensor(topic_ids, dtype=torch.long))
        self.register_buffer('outlet', torch.as_tensor(outlet_ids, dtype=torch.long))
        h = cfg.hidden
        self.text_proj = nn.Linear(self.text.shape[1], h)
        self.topic_emb = nn.Embedding(n_topics, h)
        self.outlet_emb = nn.Embedding(n_outlets, h)
        self.news_norm = nn.LayerNorm(h)
        self.self_attn = nn.MultiheadAttention(h, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.user_norm = nn.LayerNorm(h)
        self.pool = AdditiveAttention(h)
        self.dropout = nn.Dropout(cfg.dropout)
        # Wide part: linear in the point-in-time signals (a logistic-regression baseline).
        self.wide = nn.Linear(N_FEATURES, 1)
        # Deep part: learned user-article interaction; last layer starts at zero, so
        # training begins at the wide solution and only adds what generalises.
        self.deep = nn.Sequential(nn.Linear(h + 1 + N_FEATURES, 64), nn.ReLU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(64, 1))
        nn.init.zeros_(self.deep[-1].weight)
        nn.init.zeros_(self.deep[-1].bias)
        # Position-bias tower (PAL, Guo et al., RecSys 2019): in the logged data a
        # reader mostly reads what was shown at the top. This term, added to the
        # logit during training only, absorbs log P(seen | position) so that the
        # rest of the network learns relevance. It is dropped at recommendation.
        self.position_bias = nn.Embedding(MAX_POSITION, 1)
        nn.init.zeros_(self.position_bias.weight)
        # content similarity, topic affinity and outlet affinity depend on the reader;
        # freshness, topic popularity and outlet popularity are the same for everyone
        self.personal_only = cfg.personal_signals_only
        self.register_buffer('personal_mask', torch.tensor([1., 1., 0., 1., 0., 0.]))

    def news_from_parts(self, text, topic, outlet):
        x = torch.tanh(self.text_proj(text)) + self.topic_emb(topic) + self.outlet_emb(outlet)
        return self.news_norm(self.dropout(x))

    def encode_news(self, idx):
        return self.news_from_parts(self.text[idx], self.topic[idx], self.outlet[idx])

    def encode_user(self, hist, hist_mask):
        news = self.encode_news(hist)
        attended, _ = self.self_attn(news, news, news, key_padding_mask=~hist_mask)
        attended = self.user_norm(news + self.dropout(attended))
        return self.pool(attended, hist_mask)

    def forward(self, hist, hist_mask, cand, feats, positions=None, news=None):
        if self.personal_only:
            feats = feats * self.personal_mask
        user = self.encode_user(hist, hist_mask)                       # (B, H)
        news = self.encode_news(cand) if news is None else news        # (B, C, H)
        user = user.unsqueeze(1).expand_as(news)
        cosine = nn.functional.cosine_similarity(user, news, dim=-1).unsqueeze(-1)
        deep = self.deep(torch.cat([user * news, cosine, feats], dim=-1))
        logits = (self.wide(feats) + deep).squeeze(-1)
        if positions is not None:
            logits = logits + self.position_bias(positions).squeeze(-1)
        return logits


# --------------------------------------------------------------------------- #
# Training, prediction, persistence
# --------------------------------------------------------------------------- #
def _vocab(values):
    return {v: i for i, v in enumerate(sorted(set(values)))}


def _feed_losses(logits, labels, mask, kind):
    """Return (sum of losses, number of terms) for one batch.

    listwise: cross-entropy between the softmax over a feed's candidates and the
    feed's normalised positive labels (feeds without a positive are skipped).
    pointwise: binary cross-entropy per candidate.
    """
    if kind == 'pointwise':
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        return (loss * mask).sum(), mask.sum()
    positives = (labels * mask).sum(dim=1)
    keep = positives > 0
    log_probs = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
    target = (labels * mask) / positives.clamp(min=1).unsqueeze(1)
    per_feed = -(target * log_probs).sum(dim=1)
    return per_feed[keep].sum(), keep.sum()


class NeuralRecommender:
    def __init__(self, builder, cfg: NeuralConfig, topic_vocab=None, outlet_vocab=None):
        self.builder = builder
        self.cfg = cfg
        articles = builder.articles  # indexed by article_id, sorted like the vectors
        outlets = articles['outlet'].fillna('') if 'outlet' in articles else ['' for _ in range(len(articles))]
        self.topic_vocab = topic_vocab or _vocab(articles.topic)
        self.outlet_vocab = outlet_vocab or _vocab(outlets)
        topic_ids = [self.topic_vocab.get(t, 0) for t in articles.topic]
        outlet_ids = [self.outlet_vocab.get(o, 0) for o in outlets]
        torch.manual_seed(cfg.seed)
        self.net = NRMSRecommender(builder.vectors, topic_ids, outlet_ids,
                                   len(self.topic_vocab), len(self.outlet_vocab), cfg)

    def _batches(self, examples, shuffle, rng=None):
        order = np.arange(len(examples))
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order), self.cfg.batch_feeds):
            yield _collate([examples[i] for i in order[start:start + self.cfg.batch_feeds]],
                           self.cfg.max_history)

    def _loss(self, examples):
        self.net.eval()
        total, count = 0.0, 0.0
        with torch.no_grad():
            for hist, hmask, cand, cmask, feats, labels, pos in self._batches(examples, False):
                logits = self.net(hist, hmask, cand, feats, pos if self.cfg.position_debias else None)
                s, n = _feed_losses(logits, labels, cmask, self.cfg.loss)
                total += float(s)
                count += float(n)
        return total / max(count, 1.0)

    def fit(self, train, validation=None, log=print):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=cfg.learning_rate,
                                      weight_decay=cfg.weight_decay)
        history, best, waited = [], (math.inf, None, 0), 0
        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            for hist, hmask, cand, cmask, feats, labels, pos in self._batches(train, True, rng):
                optimiser.zero_grad()
                logits = self.net(hist, hmask, cand, feats, pos if cfg.position_debias else None)
                s, n = _feed_losses(logits, labels, cmask, cfg.loss)
                if n == 0:
                    continue
                loss = s / n
                if not torch.isfinite(loss):
                    raise ValueError(f'Non-finite training loss at epoch {epoch}')
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimiser.step()
            row = {'epoch': epoch, 'train_log_loss': self._loss(train)}
            if validation:
                row['validation_log_loss'] = self._loss(validation)
            history.append(row)
            monitored = row.get('validation_log_loss', row['train_log_loss'])
            log(f"epoch {epoch:3d}  train {row['train_log_loss']:.4f}"
                + (f"  validation {row['validation_log_loss']:.4f}" if validation else ''))
            if monitored < best[0] - 1e-4:
                best = (monitored, {k: v.clone() for k, v in self.net.state_dict().items()}, epoch)
                waited = 0
            else:
                waited += 1
                if validation and waited >= cfg.patience:
                    break
        if best[1] is not None:
            self.net.load_state_dict(best[1])
        labels = np.concatenate([e['labels'] for e in train])
        if cfg.loss == 'pointwise':  # constant prediction at the training prevalence
            prevalence = labels.mean()
            baseline = -float(np.mean(labels * np.log(prevalence) + (1 - labels) * np.log(1 - prevalence)))
        else:  # uniform scores over each feed with at least one positive
            baseline = float(np.mean([math.log(len(e['labels'])) for e in train if sum(e['labels'])]))
        report = {'model': 'NRMS-style neural recommender', 'config': asdict(cfg),
                  'parameters': sum(p.numel() for p in self.net.parameters() if p.requires_grad),
                  'train_feeds': len(train), 'train_rows': int(len(labels)),
                  'train_positives': int(labels.sum()), 'loss': cfg.loss,
                  'prevalence_baseline_log_loss': baseline,
                  'epochs_run': len(history), 'best_epoch': best[2],
                  'early_stopped': bool(validation) and len(history) < cfg.epochs,
                  'stop_reason': ('max_epochs' if len(history) == cfg.epochs else
                                  f'validation loss not improved by 1e-4 for {cfg.patience} epochs; '
                                  f'weights of epoch {best[2]} kept'),
                  'training_log_loss': self._loss(train), 'history': history}
        if cfg.position_debias:
            bias = self.net.position_bias.weight.detach().squeeze(-1).numpy()
            report['position_bias_relative_to_top'] = [round(float(b - bias[0]), 4) for b in bias[:26]]
        if validation:
            report['validation_feeds'] = len(validation)
            report['validation_log_loss'] = self._loss(validation)
        return report

    def score(self, examples):
        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(examples), self.cfg.batch_feeds):
                chunk = examples[start:start + self.cfg.batch_feeds]
                hist, hmask, cand, cmask, feats, _, _ = _collate(chunk, self.cfg.max_history)
                probs = torch.sigmoid(self.net(hist, hmask, cand, feats))  # no position term
                for i, e in enumerate(chunk):
                    out.append(probs[i, :len(e['cand_idx'])].numpy().astype(float))
        return out

    # -- persistence ------------------------------------------------------- #
    def save(self, path, metadata):
        buffer = io.BytesIO()
        state = {k: v for k, v in self.net.state_dict().items() if k not in ('text', 'topic', 'outlet', 'personal_mask')}
        torch.save(state, buffer)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()[:16]
        torch.save({'schema_version': 1, 'config': asdict(self.cfg), 'state_dict': state,
                    'topic_vocab': self.topic_vocab, 'outlet_vocab': self.outlet_vocab,
                    'model_version': f'nrms-{digest}', 'metadata': metadata}, path)
        return f'nrms-{digest}'

    @classmethod
    def load(cls, path, builder):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('schema_version') != 1:
            raise ValueError('Incompatible neural model format')
        model = cls(builder, NeuralConfig(**payload['config']),
                    payload['topic_vocab'], payload['outlet_vocab'])
        missing, unexpected = model.net.load_state_dict(payload['state_dict'], strict=False)
        if unexpected or set(missing) - {'text', 'topic', 'outlet', 'personal_mask'}:
            raise ValueError('Neural model weights do not match the architecture')
        return model, payload


    def score_articles(self, history_idx, text_vectors, topics, outlets, features):
        """Score articles outside the catalogue (e.g. today's RSS items) for one reader."""
        self.net.eval()
        with torch.no_grad():
            h = history_idx[-self.cfg.max_history:]
            hist = torch.zeros(1, self.cfg.max_history, dtype=torch.long)
            mask = torch.zeros(1, self.cfg.max_history, dtype=torch.bool)
            hist[0, :len(h)] = torch.tensor(h)
            mask[0, :len(h)] = True
            topic_ids = torch.tensor([self.topic_vocab.get(x, 0) for x in topics]).unsqueeze(0)
            outlet_ids = torch.tensor([self.outlet_vocab.get(x, 0) for x in outlets]).unsqueeze(0)
            text = torch.as_tensor(np.asarray(text_vectors), dtype=torch.float32).unsqueeze(0)
            news = self.net.news_from_parts(text, topic_ids, outlet_ids)
            feats = torch.as_tensor(np.asarray(features), dtype=torch.float32).unsqueeze(0)
            logits = self.net(hist, mask, None, feats, news=news)
            return torch.sigmoid(logits[0]).numpy().astype(float)


class NeuralFeedPredictor:
    """Same interface as relevance.FeedPredictor, used by the API."""

    def __init__(self, builder, model: NeuralRecommender):
        self.builder = builder
        self.model = model

    def predict_feed(self, user_id, feed_id, generated_at, candidates, split):
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError('Feed requires unique nonempty candidate IDs')
        examples, _ = build_examples(self.builder, [{'feed_id': feed_id, 'user_id': user_id,
                                     'generated_at': generated_at, 'candidates': candidates}],
                                     split, max_history=self.model.cfg.max_history)
        if not examples:
            raise ValueError(f'No eligible history for {user_id}')
        return rank_candidates(user_id, feed_id, candidates, self.model.score(examples)[0])
