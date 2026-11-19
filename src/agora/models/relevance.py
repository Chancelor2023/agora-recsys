"""Trainable relevance scorer; labels are accepted only during fitting."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

FEATURE_NAMES = ['content_similarity', 'topic_affinity', 'training_popularity']


def rank_candidates(user_id, feed_id, candidates, scores):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per candidate')
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
    return [{'user_id': user_id, 'feed_id': feed_id, 'article_id': candidates[i],
             'rank': rank, 'relevance_score': float(scores[i])}
            for rank, i in enumerate(order, 1)]


class RelevanceModel:
    def __init__(self, weights, intercept):
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)
        if self.weights.shape != (3,) or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept):
            raise ValueError('Invalid model coefficients')

    @classmethod
    def fit(cls, features, labels, regularization=1.0, seed=42):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != 3 or not np.isfinite(features).all():
            raise ValueError('Expected finite feature matrix with three columns')
        if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
            raise ValueError('Training requires matching binary labels with both classes')
        if regularization <= 0:
            raise ValueError('Regularization C must be positive')
        classifier = LogisticRegression(C=regularization, max_iter=1000, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            classifier.fit(features, labels)
        result = cls(classifier.coef_[0], classifier.intercept_[0])
        baseline = np.full(len(labels), labels.mean())
        report = {'rows': len(labels), 'positives': int(labels.sum()),
                  'prevalence_baseline_log_loss': float(log_loss(labels, baseline)),
                  'training_log_loss': float(log_loss(labels, result.predict(features))),
                  'optimizer_iterations': int(classifier.n_iter_[0]),
                  'C': regularization, 'feature_names': FEATURE_NAMES}
        return result, report

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 3 or not np.isfinite(features).all():
            raise ValueError('Invalid prediction features')
        logits = features @ self.weights + self.intercept
        return np.exp(-np.logaddexp(0, -logits))

    def save(self, path, metadata):
        payload = {'schema_version': 1, 'feature_names': FEATURE_NAMES,
                   'weights': self.weights.tolist(), 'intercept': self.intercept,
                   'metadata': metadata}
        payload['model_version'] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload['schema_version'] != 1 or payload['feature_names'] != FEATURE_NAMES:
            raise ValueError('Incompatible model format')
        return cls(payload['weights'], payload['intercept']), payload


class FeedPredictor:
    """Load once in the backend, then call predict_feed for supplied candidates."""
    def __init__(self, builder, model):
        self.builder = builder
        self.model = model

    def predict_feed(self, user_id, feed_id, generated_at, candidates, split):
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError('Feed requires unique nonempty candidate IDs')
        cutoff = self.builder.protocol.cutoff(generated_at, split)
        features, _ = self.builder.features(user_id, candidates, cutoff)
        return rank_candidates(user_id, feed_id, candidates, self.model.predict(features))
