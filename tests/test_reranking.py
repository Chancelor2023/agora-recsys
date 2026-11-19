import json
import tempfile
import unittest
from pathlib import Path

from agora.reranking import (
    ExposureState,
    FairnessAgent,
    ScrufDReranker,
    build_reranker,
    exposure_shares,
    load_feeds,
)
from agora.reranking.__main__ import main as rerank_main

ROOT = Path(__file__).resolve().parents[1]
AGENT = FairnessAgent("public_affairs", "topic", {"politics"}, target_share=0.5)


def feed(fid="f1", generated_at=""):
    return {
        "user_id": "u1",
        "feed_id": fid,
        "generated_at": generated_at,
        "candidates": [
            {"article_id": "a", "relevance_score": 0.9, "topic": "sport"},
            {"article_id": "b", "relevance_score": 0.8, "topic": "sport"},
            {"article_id": "c", "relevance_score": 0.7, "topic": "politics"},
            {"article_id": "d", "relevance_score": 0.1, "topic": "politics"},
        ],
    }


def ids(result):
    return [row["article_id"] for row in result.ranking]


class ForwardPassTests(unittest.TestCase):
    def test_first_feed_keeps_relevance_order_with_contiguous_ranks(self):
        result = ScrufDReranker([AGENT], lam=1.0, k=2).rerank(feed())
        self.assertEqual(ids(result), ["a", "b", "c", "d"])  # no history yet
        self.assertEqual([r["rank"] for r in result.ranking], [1, 2, 3, 4])
        self.assertIsNone(result.selected_agent)

    def test_state_update_changes_next_feed(self):
        rr = ScrufDReranker([AGENT], lam=1.0, k=2)
        rr.rerank(feed("f1"))
        self.assertEqual(rr.state.n_updates, 1)
        self.assertEqual(rr.state.observed_share("public_affairs"), 0.0)
        second = rr.rerank(feed("f2"))
        self.assertEqual(second.selected_agent, "public_affairs")
        self.assertEqual(ids(second)[0], "c")
        self.assertEqual(rr.state.n_updates, 2)

    def test_relevance_scores_are_preserved_separately(self):
        rr = ScrufDReranker([AGENT], lam=1.0, k=2)
        rr.rerank(feed("f1"))
        rows = rr.rerank(feed("f2")).predictions()
        by_id = {r["article_id"]: r for r in rows}
        self.assertEqual(by_id["c"]["relevance_score"], 0.7)
        self.assertGreater(by_id["c"]["reranking_score"], by_id["a"]["reranking_score"])
        self.assertEqual(set(rows[0]), {"user_id", "feed_id", "article_id", "rank",
                                        "relevance_score", "reranking_score"})

    def test_lambda_zero_is_the_ablation(self):
        rr = ScrufDReranker([AGENT], lam=0.0, k=2)
        rr.rerank(feed("f1"))
        self.assertEqual(ids(rr.rerank(feed("f2"))), ["a", "b", "c", "d"])

    def test_zero_compatibility_disables_agent(self):
        rr = ScrufDReranker([AGENT], lam=1.0, k=2)
        rr.rerank(feed("f1"))
        result = rr.rerank(feed("f2"), compatibility={"public_affairs": 0.0})
        self.assertEqual(ids(result)[:2], ["a", "b"])
        self.assertIsNone(result.selected_agent)

    def test_lottery_selects_exactly_one_agent_and_is_seeded(self):
        agents = [AGENT, FairnessAgent("environment", "topic", {"environment"}, 0.9)]
        picks = []
        for _ in range(2):
            rr = ScrufDReranker(agents, lam=1.0, k=2, seed=0)
            rr.state.update({"public_affairs": 0.0, "environment": 0.1})
            weights, _, selected = rr.allocate()
            self.assertEqual(sorted(weights.values()), [0.0, 1.0])
            picks.append(selected)
        self.assertEqual(picks[0], picks[1])

    def test_feeds_are_processed_chronologically(self):
        rr = ScrufDReranker([AGENT], lam=1.0, k=2)
        results = rr.rerank_feeds([feed("late", "2023-12-02"), feed("early", "2023-12-01")])
        self.assertEqual([r.feed_id for r in results], ["early", "late"])

    def test_invalid_feeds_raise(self):
        bad_feeds = [
            {"candidates": []},
            {"candidates": [{"article_id": "a"}]},
            {"candidates": [{"article_id": "a", "relevance_score": 1},
                            {"article_id": "a", "relevance_score": 2}]},
            {"candidates": [{"article_id": "a", "relevance_score": 1, "label": 1}]},
        ]
        for bad in bad_feeds:
            with self.assertRaises((ValueError, KeyError)):
                ScrufDReranker([AGENT]).rerank(bad)


class StateTests(unittest.TestCase):
    def test_exposure_is_position_weighted(self):
        ranked = feed()["candidates"]
        self.assertEqual(exposure_shares(ranked, [AGENT], k=2)["public_affairs"], 0.0)
        share = exposure_shares(ranked, [AGENT], k=3)["public_affairs"]
        self.assertTrue(0.0 < share < 1 / 3)

    def test_window_and_round_trip(self):
        state = ExposureState(window=2)
        for value in (0.1, 0.2, 0.3):
            state.update({"public_affairs": value})
        self.assertAlmostEqual(state.observed_share("public_affairs"), 0.25)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state.save(path)
            self.assertEqual(ExposureState.load(path).to_dict(), state.to_dict())


class IntegrationTests(unittest.TestCase):
    def test_config_and_flat_rows(self):
        rr = build_reranker(json.loads((ROOT / "configs/reranking_agents.json").read_text()))
        self.assertEqual(rr.allocation, "lottery")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.json"
            rows = [{"user_id": "u1", "feed_id": "f1", "article_id": x,
                     "relevance_score": s, "topic": "sport"} for x, s in (("a", 1), ("b", 2))]
            path.write_text(json.dumps({"predictions": rows}))
            feeds = load_feeds(path)
        self.assertEqual(len(feeds), 1)
        self.assertEqual(len(feeds[0]["candidates"]), 2)

    def test_cli_output_is_accepted_by_evaluation_harness(self):
        from agora.evaluation.harness import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preds.json"
            result = rerank_main(["--input", str(ROOT / "examples/reranking_mock_input.json"),
                                  "--config", str(ROOT / "configs/reranking_agents.json"),
                                  "--output", str(out)])
            self.assertEqual(result["final_state"]["n_updates"], 3)
            predictions = json.loads(out.read_text())["predictions"]
        labels = json.loads((ROOT / "examples/reranking_mock_labels.json").read_text())["labels"]
        metrics = evaluate(predictions, labels, 10, 20)
        self.assertEqual(metrics["total_feeds"], 3)
        self.assertIsNotNone(metrics["metrics"]["NDCG@10"])


if __name__ == "__main__":
    unittest.main()
