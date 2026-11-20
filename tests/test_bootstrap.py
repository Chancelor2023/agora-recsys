"""Paired bootstrap of ranking differences (synthetic, hand-checkable cases)."""
import json
import unittest

from agora.evaluation import evaluate
from agora.evaluation.bootstrap import compare, main, paired_bootstrap


def _predictions(order_by_feed, users):
    return [{"user_id": users[feed], "feed_id": feed, "article_id": article,
             "rank": rank, "relevance_score": float(-rank)}
            for feed, order in order_by_feed.items() for rank, article in enumerate(order, 1)]


def _setup():
    """6 feeds of 3 readers; one feed without positive; article 'p' is the read one."""
    users = {"f1": "u1", "f2": "u1", "f3": "u2", "f4": "u2", "f5": "u3", "f6": "u3"}
    candidates = ["p", "x", "y", "z"]
    labels = [{"feed_id": f, "article_id": a, "label": int(a == "p" and f != "f6")}
              for f in users for a in candidates]
    good = {f: ["p", "x", "y", "z"] for f in users}       # read article first
    bad = {f: ["x", "y", "z", "p"] for f in users}        # read article last
    return users, labels, good, bad


class PairedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.users, self.labels, self.good, self.bad = _setup()

    def evaluate(self, orders):
        return evaluate(_predictions(orders, self.users), self.labels, 10, 20)

    def test_identical_rankings_give_zero_interval(self):
        a = self.evaluate(self.good)
        out = paired_bootstrap(a, a, resamples=200)
        self.assertEqual((out["difference_a_minus_b"], out["ci_low"], out["ci_high"]), (0.0, 0.0, 0.0))
        self.assertEqual(out["evaluated_feeds"], 5)   # f6 has no positive: excluded
        self.assertEqual(out["units"], 3)

    def test_uniform_gain_is_recovered_exactly(self):
        # NDCG@10 = 1 with the read article first, 1/log2(5) with it fourth.
        out = paired_bootstrap(self.evaluate(self.good), self.evaluate(self.bad), resamples=200)
        gain = 1 - 1 / 2.321928094887362
        self.assertAlmostEqual(out["difference_a_minus_b"], gain)
        self.assertAlmostEqual(out["ci_low"], gain)
        self.assertAlmostEqual(out["ci_high"], gain)
        self.assertEqual(out["share_of_resamples_above_zero"], 1.0)

    def test_deterministic_for_a_seed(self):
        mixed = dict(self.good, f3=self.bad["f3"], f5=self.bad["f5"])
        a, b = self.evaluate(mixed), self.evaluate(self.bad)
        first = paired_bootstrap(a, b, resamples=300, seed=7)
        self.assertEqual(first, paired_bootstrap(a, b, resamples=300, seed=7))
        self.assertLessEqual(first["ci_low"], first["difference_a_minus_b"])
        self.assertGreaterEqual(first["ci_high"], first["difference_a_minus_b"])

    def test_user_unit_keeps_a_readers_feeds_together(self):
        # Gain only for u1 (two feeds): user resampling draws 0, 2, 4 or 6 such feeds.
        mixed = dict(self.bad, f1=self.good["f1"], f2=self.good["f2"])
        out = paired_bootstrap(self.evaluate(mixed), self.evaluate(self.bad), unit="user",
                               resamples=500, seed=1)
        self.assertEqual(out["units"], 3)
        feed_level = paired_bootstrap(self.evaluate(mixed), self.evaluate(self.bad), unit="feed",
                                      resamples=500, seed=1)
        self.assertEqual(feed_level["units"], 5)
        self.assertAlmostEqual(out["difference_a_minus_b"], feed_level["difference_a_minus_b"])

    def test_rejects_different_feed_sets(self):
        a = self.evaluate(self.good)
        other_labels = [row for row in self.labels if row["feed_id"] != "f1"]
        orders = {f: o for f, o in self.good.items() if f != "f1"}
        b = evaluate(_predictions(orders, self.users), other_labels, 10, 20)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b)

    def test_rejects_invalid_options(self):
        a = self.evaluate(self.good)
        for options in ({"unit": "day"}, {"resamples": 0}, {"alpha": 1.5}, {"metric": "MRR"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                paired_bootstrap(a, a, **options)

    def test_compare_reports_exclusion_counts(self):
        out = compare(_predictions(self.good, self.users), _predictions(self.bad, self.users),
                      self.labels, resamples=100)
        self.assertEqual((out["total_feeds"], out["evaluated_feeds"], out["excluded_zero_positive_feeds"]),
                         (6, 5, 1))
        self.assertEqual([c["metric"] for c in out["comparisons"]], ["NDCG@10", "Recall@20"])

    def test_command_line(self):
        import tempfile
        from pathlib import Path
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text(json.dumps({"predictions": _predictions(self.good, self.users)}))
            (root / "b.json").write_text(json.dumps({"predictions": _predictions(self.bad, self.users)}))
            (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in self.labels))
            with redirect_stdout(StringIO()):
                main(["--labels", str(root / "labels.jsonl"), "--a", str(root / "a.json"),
                      "--b", str(root / "b.json"), "--resamples", "50", "--output", str(root / "out.json")])
            report = json.loads((root / "out.json").read_text())
        self.assertEqual(report["evaluated_feeds"], 5)
        self.assertGreater(report["comparisons"][0]["difference_a_minus_b"], 0)


if __name__ == "__main__":
    unittest.main()
