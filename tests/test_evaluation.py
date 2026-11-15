"""Hand-calculated metrics and evaluation-integrity regression checks."""
import copy
import json
from math import log2
from pathlib import Path
import unittest
from agora.evaluation import evaluate, ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_perfect_and_short_rankings(self):
        self.assertEqual(ndcg_at_k([1, 1, 0]), 1)
        self.assertEqual(recall_at_k([1, 0]), 1)

    def test_hand_calculation(self):
        labels = [1 if rank in (1, 11, 21) else 0 for rank in range(1, 27)]
        self.assertAlmostEqual(ndcg_at_k(labels), 1 / (1 + 1 / log2(3) + .5))
        self.assertAlmostEqual(recall_at_k(labels), 2 / 3)

    def test_cutoff_boundaries(self):
        labels = [int(rank in (10, 20, 21)) for rank in range(1, 27)]
        self.assertAlmostEqual(ndcg_at_k(labels), (1 / log2(11)) / (1 + 1 / log2(3) + .5))
        self.assertAlmostEqual(recall_at_k(labels), 2 / 3)

    def test_no_positive(self):
        self.assertIsNone(ndcg_at_k([0, 0]))
        self.assertIsNone(recall_at_k([]))

    def test_invalid_values(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(k=value), self.assertRaises(ValueError):
                ndcg_at_k([1], value)
        for labels in ([2], [-1], [.5], [True]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                recall_at_k(labels)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.predictions = json.loads((ROOT / "examples/mock_predictions.json").read_text())["predictions"]
        self.labels = json.loads((ROOT / "examples/mock_labels.json").read_text())["labels"]

    def test_fixture_and_unordered_rows(self):
        result = evaluate(list(reversed(self.predictions)), self.labels)
        self.assertEqual(result["total_feeds"], 3)
        self.assertEqual(result["evaluated_feeds"], 2)
        self.assertEqual(result["excluded_zero_positive_feeds"], 1)
        expected = (1 / (1 + 1 / log2(3) + .5) + 1) / 2
        self.assertAlmostEqual(result["metrics"]["NDCG@10"], expected)
        self.assertAlmostEqual(result["metrics"]["Recall@20"], 5 / 6)

    def test_all_zero_returns_null_not_misleading_zero(self):
        labels = [dict(row, label=0) for row in self.labels]
        result = evaluate(self.predictions, labels)
        self.assertEqual(result["evaluated_feeds"], 0)
        self.assertTrue(all(value is None for value in result["metrics"].values()))

    def test_reject_duplicate_prediction(self):
        with self.assertRaises(ValueError):
            evaluate(self.predictions + [self.predictions[0]], self.labels)

    def test_reject_missing_candidate_or_feed(self):
        for predictions in (self.predictions[1:],
                            [r for r in self.predictions if r["feed_id"] != "mock-empty"]):
            with self.subTest(), self.assertRaises(ValueError):
                evaluate(predictions, self.labels)

    def test_reject_duplicate_or_missing_labels(self):
        for labels in (self.labels + [self.labels[0]], self.labels[1:]):
            with self.subTest(), self.assertRaises(ValueError):
                evaluate(self.predictions, labels)

    def test_reject_bad_ranks_user_scores_and_leaked_labels(self):
        for change in ({"rank": 2}, {"rank": True}, {"user_id": "other"},
                       {"relevance_score": float("nan")}, {"label": 1}):
            rows = copy.deepcopy(self.predictions)
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate(rows, self.labels)

    def test_explicit_rank_is_authoritative_even_with_equal_scores(self):
        rows = [dict(row, relevance_score=0) for row in self.predictions]
        self.assertEqual(evaluate(rows, self.labels), evaluate(self.predictions, self.labels))

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [])


if __name__ == "__main__":
    unittest.main()
