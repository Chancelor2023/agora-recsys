"""Trainable relevance scorer; labels are accepted only during fitting.

The scorer is an L2-regularised logistic regression trained epoch by epoch
(full-batch Adam), so that training and validation log loss can be tracked,
plotted and used for early stopping. Validation labels are used only for the
loss curve and to select the best epoch; test labels are never used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURE_NAMES = ['content_similarity', 'topic_affinity', 'training_popularity']
EPS = 1e-12


def log_loss(labels, probabilities):
    """Binary cross-entropy, clipped for numerical stability."""
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _check_xy(features, labels, name):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[1] != 3 or not np.isfinite(features).all():
        raise ValueError(f'Expected finite {name} feature matrix with three columns')
    if labels.shape != (len(features),) or not set(labels.tolist()) <= {0, 1}:
        raise ValueError(f'Expected matching binary {name} labels')
    return features, labels.astype(float)


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
    def fit(cls, features, labels, regularization=1.0, seed=42, validation=None,
            epochs=300, learning_rate=0.05, patience=30, min_delta=1e-5):
        """Train epoch by epoch; return (model, report with the loss history).

        ``validation`` = (features, labels) from the validation split. When it is
        given, the weights of the epoch with the lowest validation log loss are
        kept and training stops after ``patience`` epochs without improvement.
        """
        features, labels = _check_xy(features, labels, 'training')
        if set(labels.tolist()) != {0.0, 1.0}:
            raise ValueError('Training requires matching binary labels with both classes')
        if regularization <= 0:
            raise ValueError('Regularization C must be positive')
        if epochs < 1 or learning_rate <= 0 or patience < 1:
            raise ValueError('epochs, learning_rate and patience must be positive')
        if validation is not None:
            val_x, val_y = _check_xy(*validation, 'validation')
            if len(val_y) == 0:
                validation = None

        n = len(labels)
        l2 = 1.0 / (regularization * n)  # same scaling as scikit-learn's C
        rng = np.random.default_rng(seed)
        params = rng.normal(0.0, 0.01, size=4)  # 3 weights + intercept
        m, v = np.zeros(4), np.zeros(4)
        beta1, beta2 = 0.9, 0.999
        design = np.column_stack([features, np.ones(n)])

        def probabilities(x, p):
            return np.exp(-np.logaddexp(0, -(x @ p[:3] + p[3])))

        history = []
        best = (np.inf, params.copy(), 0)
        waited = 0
        for epoch in range(1, epochs + 1):
            prob = probabilities(features, params)
            gradient = design.T @ (prob - labels) / n
            gradient[:3] += l2 * params[:3]
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient ** 2
            m_hat, v_hat = m / (1 - beta1 ** epoch), v / (1 - beta2 ** epoch)
            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            if not np.isfinite(params).all():
                raise ValueError(f'Training diverged (non-finite weights) at epoch {epoch}')

            row = {'epoch': epoch, 'train_log_loss': log_loss(labels, probabilities(features, params))}
            if validation is not None:
                row['validation_log_loss'] = log_loss(val_y, probabilities(val_x, params))
                monitored = row['validation_log_loss']
            else:
                monitored = row['train_log_loss']
            history.append(row)
            if not np.isfinite(monitored):
                raise ValueError(f'Non-finite loss at epoch {epoch}')
            improved_enough = monitored < best[0] - min_delta
            if monitored < best[0]:
                best = (monitored, params.copy(), epoch)
            if improved_enough:
                waited = 0
            else:
                waited += 1
                if validation is not None and waited >= patience:
                    break

        chosen = best[1] if validation is not None else params
        result = cls(chosen[:3], chosen[3])
        baseline = np.full(n, labels.mean())
        report = {'rows': n, 'positives': int(labels.sum()),
                  'prevalence_baseline_log_loss': log_loss(labels, baseline),
                  'training_log_loss': log_loss(labels, result.predict(features)),
                  'optimizer_iterations': len(history),
                  'epochs_run': len(history),
                  'best_epoch': best[2] if validation is not None else len(history),
                  'early_stopped': validation is not None and len(history) < epochs,
                  # 'converged': stopped because the monitored loss improved by less than
                  # min_delta for `patience` epochs, even if the last epoch is the best one.
                  'stop_reason': ('max_epochs' if len(history) == epochs else
                                  f'converged: improvement < {min_delta} for {patience} epochs'
                                  + ('' if best[2] < len(history) else ' (last epoch is the best)')),
                  'optimizer': 'full-batch Adam', 'learning_rate': learning_rate,
                  'patience': patience, 'C': regularization, 'feature_names': FEATURE_NAMES,
                  'history': history}
        if validation is not None:
            report.update({
                'validation_rows': len(val_y), 'validation_positives': int(val_y.sum()),
                'validation_log_loss': log_loss(val_y, result.predict(val_x)),
                'validation_prevalence_baseline_log_loss':
                    log_loss(val_y, np.full(len(val_y), labels.mean())),
            })
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
