import json
import threading

import pandas as pd
import pytest

from agora.data import pipeline
from agora.data.idea import load_idea
from agora.data.storage import InteractionStore


def _inputs():
    reads = pd.DataFrame({"user_id": ["u1"] * 6,
                          "article_id": [f"a{i}" for i in range(6)],
                          "read_at": [f"2023-11-24 1{i}:00:00" for i in range(6)]})
    feeds = pd.DataFrame({"feed_id": ["f1"], "user_id": ["u1"], "generated_at": ["2023-11-30 09:00:00"],
                          "candidates": [["a1", "a2", "a9"]]})
    articles = pd.DataFrame({"article_id": [f"a{i}" for i in range(10)], "title": "t", "lead": "l",
                             "topic": "politics", "outlet": "o", "published_at": "2023-11-20T08:00:00Z"})
    return feeds, reads, articles


def test_build_reports_counts_when_catalogue_is_complete(tmp_path):
    feeds, reads, articles = _inputs()
    pipeline.run(feeds, reads, tmp_path, articles=articles)
    report = json.loads((tmp_path / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["candidates_missing_from_catalogue"] == 0
    assert report["reads_outside_experiment"] == 0
    assert report["articles_missing_values"]["lead"] == 0


def test_build_stops_on_missing_catalogue_item_and_exports_nothing(tmp_path):
    feeds, reads, articles = _inputs()
    with pytest.raises(SystemExit):
        pipeline.run(feeds, reads, tmp_path, articles=articles[articles.article_id != "a9"])
    assert not (tmp_path / "articles.json").exists()
    assert not list(tmp_path.glob("feeds_*"))


def test_idea_feed_id_collision_is_an_error(tmp_path):
    rec = {"informfully_id": "u1", "createdAt": "2023-11-30T09:00:00.000Z", "articleId": "a1",
           "prediction": 1.0, "recommendationAlgorithm": "x", "experimental_week": 1}
    other = {**rec, "createdAt": "2023-11-30 09:00:00.000", "articleId": "a2"}  # same digits
    (tmp_path / "recommendations.json").write_text(json.dumps([rec, other]))
    (tmp_path / "interactions.json").write_text("[]")
    (tmp_path / "articles.json").write_text("[]")
    with pytest.raises(ValueError, match="feed_id collision"):
        load_idea(tmp_path)


def test_interaction_store_is_safe_across_threads(tmp_path):
    store = InteractionStore(tmp_path / "agora.db")

    def write(t):
        for i in range(50):
            store.insert({"event_id": f"{t}-{i}", "event_type": "click", "user_id": "u1",
                          "article_id": "a1", "feed_id": "f1", "timestamp": "2026-11-20T10:00:00Z",
                          "displayed_rank": 1, "model_version": "test"})
    threads = [threading.Thread(target=write, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 400
