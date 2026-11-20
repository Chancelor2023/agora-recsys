import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agora.data.live import collect, parse_feed
from agora.models.neural import NeuralConfig, NeuralRecommender, build_examples
from agora.models.live import live_features
from agora.profiles import ProfileBuilder, TemporalProtocol

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)


class LiveIngestionTests(unittest.TestCase):
    def test_parse_bbc_rss(self):
        items = parse_feed((FIXTURES / "rss_bbc_sample.xml").read_bytes(),
                           {"url": "x", "outlet": "BBC", "topic": "uk_news"}, NOW)
        self.assertEqual(len(items), 3)
        first = items[0]
        self.assertEqual(first["published_at"], "2026-11-19T09:12:00Z")
        self.assertEqual(first["topic"], "uk_news")
        self.assertTrue(first["image"].startswith("https://"))
        self.assertTrue(first["article_id"].startswith("live-"))

    def test_guardian_topics_inferred_and_html_stripped(self):
        items = parse_feed((FIXTURES / "rss_guardian_sample.xml").read_bytes(),
                           {"url": "x", "outlet": "GuardianInt"}, NOW)
        self.assertEqual([i["topic"] for i in items], ["environment", "politics"])
        self.assertNotIn("<", items[0]["lead"])

    def test_collect_skips_broken_feed_and_old_items(self):
        def fetcher(url):
            if url == "broken":
                raise OSError("unreachable")
            return (FIXTURES / "rss_bbc_sample.xml").read_bytes()
        logs = []
        items = collect([{"url": "broken", "outlet": "BBC"}, {"url": "ok", "outlet": "BBC"}],
                        max_age_hours=12, fetcher=fetcher, now=NOW, log=logs.append)
        self.assertEqual(len(items), 2)  # the 21:55 item of the day before is older than 12 h
        self.assertTrue(any("skipped broken" in line for line in logs))


def _builder():
    protocol = TemporalProtocol.from_config({
        'train_start': '2023-11-23T00:00:00Z', 'train_end': '2023-11-30T00:00:00Z',
        'validation_end': '2023-12-03T00:00:00Z', 'test_end': '2023-12-08T00:00:00Z',
        'min_train_articles': 2})
    rng = np.random.default_rng(0)
    ids = [f'a{i:02d}' for i in range(20)]
    articles = pd.DataFrame({'article_id': ids, 'topic': ['politics', 'sport'] * 10,
                             'outlet': ['BBC', 'GuardianInt'] * 10,
                             'published_at': ['2023-11-23T08:00:00Z'] * 20})
    reads = pd.DataFrame([{'user_id': u, 'article_id': ids[j], 'ts': f'2023-11-{23 + d}T10:00:00Z'}
                          for u, start in (('u', 0), ('v', 1)) for d, j in enumerate(range(start, 12, 2))])
    return ProfileBuilder(reads, articles, rng.normal(size=(20, 8)), protocol), ids


class NeuralModelTests(unittest.TestCase):
    def setUp(self):
        self.builder, self.ids = _builder()
        feeds, rows = [], []
        for day in (26, 27, 28):
            for user in ('u', 'v'):
                fid = f'{user}-{day}'
                feeds.append({'feed_id': fid, 'user_id': user, 'generated_at': f'2023-11-{day}T02:00:00Z',
                              'candidates': self.ids[12:20]})
                rows += [{'feed_id': fid, 'article_id': a, 'label': int((a[-1] in '02468') == (user == 'u'))}
                         for a in self.ids[12:20]]
        self.examples, _ = build_examples(self.builder, feeds, 'train', pd.DataFrame(rows))

    def test_training_history_saved_and_reloaded(self):
        cfg = NeuralConfig(hidden=16, heads=2, epochs=5, batch_feeds=4, learning_rate=1e-2)
        model = NeuralRecommender(self.builder, cfg)
        report = model.fit(self.examples, self.examples, log=lambda _: None)
        self.assertTrue(all(np.isfinite(r['validation_log_loss']) for r in report['history']))
        self.assertIn('position_bias_relative_to_top', report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'neural.pt'
            model.save(path, {'binding': {}})
            restored, payload = NeuralRecommender.load(path, self.builder)
        np.testing.assert_allclose(model.score(self.examples)[0], restored.score(self.examples)[0], rtol=1e-6)
        self.assertTrue(payload['model_version'].startswith('nrms-'))

    def test_scores_articles_outside_the_catalogue(self):
        model = NeuralRecommender(self.builder, NeuralConfig(hidden=16, heads=2))
        live = pd.DataFrame({'topic': ['politics', 'sport', 'unknown-topic'],
                             'outlet': ['BBC', 'BBC', 'NewOutlet'],
                             'published_at': ['2026-11-19T09:00:00Z', '2026-11-18T09:00:00Z', 'bad']})
        vectors = np.random.default_rng(1).normal(size=(3, 8))
        features, history = live_features(self.builder, 'u', live, vectors, now=NOW)
        self.assertEqual(features.shape, (3, 6))
        self.assertGreater(features[0, 2], features[1, 2])  # fresher article
        scores = model.score_articles(history, vectors, live.topic.tolist(), live.outlet.tolist(), features)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
