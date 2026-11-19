import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from agora.profiles import ProfileBuilder, TemporalProtocol
from agora.models.relevance import RelevanceModel, FeedPredictor, rank_candidates
from agora.models.io import read_records, load_feeds


class RelevanceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = TemporalProtocol.from_config({
            'train_start': '2023-11-23T00:00:00Z', 'train_end': '2023-11-30T00:00:00Z',
            'validation_end': '2023-12-03T00:00:00Z', 'test_end': '2023-12-08T00:00:00Z',
            'min_train_articles': 2})
        self.articles = pd.DataFrame({'article_id': ['a', 'b', 'c'], 'topic': ['science', 'sport', 'science']})
        self.vectors = np.array([[1, 0], [0, 1], [1, 0]])
        self.reads = pd.DataFrame([
            {'user_id': 'u', 'article_id': 'a', 'ts': '2023-11-23T01:00:00Z'},
            {'user_id': 'u', 'article_id': 'b', 'ts': '2023-11-25T01:00:00Z'},
            {'user_id': 'v', 'article_id': 'c', 'ts': '2023-11-26T01:00:00Z'}])
        self.builder = ProfileBuilder(self.reads, self.articles, self.vectors, self.protocol)

    def test_training_excludes_future_history_and_popularity(self):
        features, count = self.builder.features('u', ['a', 'b', 'c'], pd.Timestamp('2023-11-24T00:00Z'))
        np.testing.assert_allclose(features, [[1, 1, 1], [0, 0, 0], [1, 1, 0]])
        self.assertEqual(count, 1)

    def test_exact_timestamp_excluded(self):
        _, count = self.builder.features('u', ['a'], pd.Timestamp('2023-11-25T01:00Z'))
        self.assertEqual(count, 1)

    def test_validation_and_test_freeze_same_cutoff(self):
        self.assertEqual(self.protocol.cutoff('2023-11-30T12:00Z', 'validation'),
                         self.protocol.cutoff('2023-12-06T12:00Z', 'test'))

    def test_rejects_future_reads(self):
        self.reads.loc[0, 'ts'] = '2023-12-01T00:00Z'
        with self.assertRaises(ValueError):
            ProfileBuilder(self.reads, self.articles, self.vectors, self.protocol)

    def test_rejects_naive_timestamp(self):
        with self.assertRaises(ValueError):
            self.protocol.cutoff('2023-11-24', 'train')

    def test_repeated_reads_do_not_inflate_eligibility(self):
        reads = pd.concat([self.reads.iloc[[0]], self.reads.iloc[[0]]])
        builder = ProfileBuilder(reads, self.articles, self.vectors, self.protocol)
        self.assertNotIn('u', builder.eligible)

    def test_empty_history_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.features('u', ['a'], self.protocol.train_start)

    def test_full_ranking_and_tie_break(self):
        rows = rank_candidates('u', 'f', ['c', 'a', 'b'], [0.5, 0.5, 0.8])
        self.assertEqual([r['article_id'] for r in rows], ['b', 'a', 'c'])
        self.assertEqual([r['rank'] for r in rows], [1, 2, 3])

    def test_nonfinite_score_rejected(self):
        with self.assertRaises(ValueError):
            rank_candidates('u', 'f', ['a'], [np.nan])

    def test_model_fit_reload(self):
        x = np.array([[1, 1, 0], [0, 0, 0], [.8, 1, .2], [.1, 0, .1]])
        model, report = RelevanceModel.fit(x, [1, 0, 1, 0])
        self.assertLess(report['training_log_loss'], report['prevalence_baseline_log_loss'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            model.save(path, {})
            restored, _ = RelevanceModel.load(path)
            np.testing.assert_array_equal(model.predict(x), restored.predict(x))

    def _separable(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, size=(n, 3))
        y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.2, n) > 0.8).astype(int)
        return x, y

    def test_epoch_history_decreases_and_is_finite(self):
        x, y = self._separable()
        _, report = RelevanceModel.fit(x, y, epochs=120)
        losses = [row['train_log_loss'] for row in report['history']]
        self.assertEqual(len(losses), 120)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(losses[-1], losses[0])
        self.assertLess(report['training_log_loss'], report['prevalence_baseline_log_loss'])

    def test_validation_curve_and_best_epoch_kept(self):
        x, y = self._separable(seed=1)
        vx, vy = self._separable(n=80, seed=2)
        model, report = RelevanceModel.fit(x, y, validation=(vx, vy), epochs=400, patience=15)
        self.assertIn('validation_log_loss', report['history'][0])
        best = min(row['validation_log_loss'] for row in report['history'])
        self.assertAlmostEqual(report['validation_log_loss'], best, places=6)
        self.assertLessEqual(report['best_epoch'], report['epochs_run'])

    def test_training_is_deterministic(self):
        x, y = self._separable()
        a, _ = RelevanceModel.fit(x, y, epochs=50, seed=7)
        b, _ = RelevanceModel.fit(x, y, epochs=50, seed=7)
        np.testing.assert_array_equal(a.weights, b.weights)

    def test_single_class_rejected(self):
        with self.assertRaises(ValueError):
            RelevanceModel.fit(np.ones((2, 3)), [0, 0])

    def test_missing_candidate_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.features('u', ['unknown'], self.protocol.train_end)

    def test_jsonl_with_json_extension_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.json'
            path.write_text('{"label": 1}\n{"label": 0}\n')
            self.assertEqual(len(read_records(path)), 2)

    def test_duplicate_candidates_rejected(self):
        predictor = FeedPredictor(self.builder, RelevanceModel([1, 1, 1], 0))
        with self.assertRaises(ValueError):
            predictor.predict_feed('u', 'f', '2023-11-30T12:00Z', ['a', 'a'], 'validation')

    def test_feed_outside_split_rejected(self):
        with self.assertRaises(ValueError):
            self.protocol.cutoff('2023-12-03T00:00Z', 'validation')


if __name__ == '__main__':
    unittest.main()
