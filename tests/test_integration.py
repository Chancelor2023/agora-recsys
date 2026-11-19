"""Regression tests for the shared config and evaluation boundaries."""
import re
import tempfile
from pathlib import Path
import unittest
from agora.models.config import load_model_config
from agora.evaluation.__main__ import _load
# from agora.evaluation.coverage import coverage_at_k

ROOT = Path(__file__).resolve().parents[1]


class SharedIntegrationTests(unittest.TestCase):
    def config(self, timezone):
        text = (ROOT / 'configs/experiment.toml').read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.toml'
            text = re.sub(r'(?m)^source_timezone = ".*"$', f'source_timezone = "{timezone}"', text)
            path.write_text(text)
            return load_model_config(path)

    def test_unverified_timezone_rejected(self):
        with self.assertRaises(ValueError):
            self.config('UNVERIFIED')

    def test_local_midnight_is_converted(self):
        config = self.config('Europe/Paris')
        self.assertEqual(config['train_start'], '2023-11-22T23:00:00+00:00')
        self.assertEqual(config['min_train_articles'], 5)

    def test_invalid_timezone_rejected(self):
        with self.assertRaises(ValueError):
            self.config('Invalid/Timezone')

    def test_evaluation_loads_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.json'
            path.write_text('{"label": 1}\n{"label": 0}\n')
            self.assertEqual(_load(path, 'labels'), [{'label': 1}, {'label': 0}])

    # def test_coverage_denominators(self):
    #     rows = [{'feed_id': f, 'article_id': a, 'rank': i}
    #             for f in ['x', 'y'] for i, a in enumerate(['a', 'b', 'c'], 1)]
    #     articles = [{'article_id': a, 'topic': t}
    #                 for a, t in [('a', 'science'), ('b', 'science'), ('c', 'sport'), ('unused', 'politics')]]
    #     result = coverage_at_k(rows, articles, 2)
    #     self.assertEqual(result['topic_coverage_macro_by_feed'], 0.5)
    #     self.assertEqual(result['catalogue_coverage'], 2 / 3)

    # def test_coverage_missing_metadata_rejected(self):
    #     with self.assertRaises(ValueError):
    #         coverage_at_k([{'feed_id': 'f', 'article_id': 'a', 'rank': 1}], [], 1)
