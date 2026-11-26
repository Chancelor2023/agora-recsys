import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from agora.api.main import create_app
from agora.api.service import RecommenderService
from agora.data.storage import InteractionStore
from agora.models.relevance import FeedPredictor, RelevanceModel
from agora.profiles import ProfileBuilder, TemporalProtocol
from agora.reranking import FairnessAgent, ScrufDReranker


client = TestClient(create_app(mock=True, store=InteractionStore(":memory:")))


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations():
    response = client.get(
        "/recommend",
        params={
            "user_id": "test-user",
            "feed_id": "test-feed",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "test-user"
    assert data["feed_id"] == "test-feed"
    assert data["model_version"] == "mock-v1"
    assert data["is_mock"] is True

    assert len(data["items"]) == 2
    assert data["items"][0]["article_id"] == "demo-a"
    assert data["items"][0]["rank"] == 1
    assert data["items"][1]["article_id"] == "demo-b"
    assert data["items"][1]["rank"] == 2

    for item in data["items"]:
        assert "title" in item
        assert "lead" in item
        assert "outlet" in item
        assert "published_at" in item


def test_recommendations_requires_parameters():
    response = client.get("/recommend")

    assert response.status_code == 422


def test_log_interaction_is_persisted():
    event = {"user_id": "u", "feed_id": "f", "article_id": "a", "event_type": "click",
             "displayed_rank": 1, "model_version": "mock-v1", "event_id": "e-1"}
    first = client.post("/log_interaction", json=event)
    assert first.status_code == 200 and first.json()["stored"] is True
    assert client.post("/log_interaction", json=event).json()["stored"] is False  # same id
    rows = client.get("/interactions", params={"user_id": "u"}).json()
    assert rows[0]["event_id"] == "e-1" and rows[0]["event_type"] == "click"


def test_log_interaction_rejects_bad_event():
    bad = {"user_id": "u", "feed_id": "f", "article_id": "a", "event_type": "like",
           "displayed_rank": 0, "model_version": "v"}
    assert client.post("/log_interaction", json=bad).status_code == 422


def _tiny_service():
    protocol = TemporalProtocol.from_config({
        'train_start': '2023-11-23T00:00:00Z', 'train_end': '2023-11-30T00:00:00Z',
        'validation_end': '2023-12-03T00:00:00Z', 'test_end': '2023-12-08T00:00:00Z',
        'min_train_articles': 2})
    articles = pd.DataFrame({
        'article_id': ['a', 'b', 'c'], 'topic': ['sport', 'politics', 'sport'],
        'title': ['A', 'B', 'C'], 'lead': ['la', 'lb', 'lc'], 'outlet': ['BBC'] * 3,
        'published_at': ['2023-11-29T08:00:00Z'] * 3})
    reads = pd.DataFrame([{'user_id': 'u', 'article_id': 'a', 'ts': '2023-11-23T01:00:00Z'},
                          {'user_id': 'u', 'article_id': 'c', 'ts': '2023-11-24T01:00:00Z'}])
    builder = ProfileBuilder(reads, articles, np.eye(3), protocol)
    feeds = {'f1': {'feed_id': 'f1', 'user_id': 'u', 'generated_at': '2023-12-01T02:00:00Z',
                    'candidates': ['a', 'b', 'c'], 'split': 'validation'}}
    reranker = ScrufDReranker([FairnessAgent('public_affairs', 'topic', {'politics'}, 0.5)], lam=1.0, k=2)
    return RecommenderService(FeedPredictor(builder, RelevanceModel([1, 1, 1], 0)),
                              articles, feeds, reranker, 'test-version')


def test_real_service_scores_and_reranks():
    real = TestClient(create_app(service=_tiny_service(), store=InteractionStore(":memory:")))
    assert real.get("/users").json() == ["u"]
    assert real.get("/users/u/feeds").json()[0]["feed_id"] == "f1"
    data = real.get("/recommend", params={"user_id": "u", "feed_id": "f1"}).json()
    assert data["is_mock"] is False and data["model_version"] == "test-version"
    assert [i["rank"] for i in data["items"]] == [1, 2, 3]
    assert {i["article_id"] for i in data["items"]} == {"a", "b", "c"}
    assert all(i["relevance_score"] is not None and i["topic"] for i in data["items"])
    assert data["latency_ms"] is not None
    assert real.get("/recommend", params={"user_id": "x", "feed_id": "f1"}).status_code == 422
    assert real.get("/recommend", params={"user_id": "u", "feed_id": "nope"}).status_code == 404


def test_reader_profile_and_window_setting():
    service = _tiny_service()
    service.rerank_config = {"lambda": 1.0, "k": 2, "agents": [
        {"name": "public_affairs", "attribute": "topic", "protected_values": ["politics"], "target_share": 0.5}]}
    real = TestClient(create_app(service=service, store=InteractionStore(":memory:")))
    profile = real.get("/users/u/profile").json()
    assert profile["reads"] == 2 and profile["topics"] == {"sport": 2}
    assert real.get("/users/nope/profile").status_code == 404
    # lam=0: relevance order only, served by a separate reranker; the default one is untouched
    plain = real.get("/recommend", params={"user_id": "u", "feed_id": "f1", "lam": 0}).json()
    assert [i["reranking_score"] for i in plain["items"]] == sorted(
        (i["reranking_score"] for i in plain["items"]), reverse=True)
    assert service.reranker.state.n_updates == 0
    assert real.get("/recommend", params={"user_id": "u", "feed_id": "f1", "lam": 5}).status_code == 422


def test_mock_profile_is_empty():
    assert client.get("/users/demo-user/profile").json()["topics"] == {}


def test_live_shortlist_keeps_protected_topics():
    from agora.api.service import _personal_pool
    service = _tiny_service()
    builder = service.predictor.builder
    live = pd.DataFrame({"topic": ["sport"] * 6 + ["politics"] * 3})
    vectors = np.array([[1, 0, 0]] * 6 + [[0, 1, 0]] * 3, dtype=float)
    plain = _personal_pool(builder, "u", live, vectors, 4)
    kept = _personal_pool(builder, "u", live, vectors, 4, ["politics"], 3)
    assert sum(live.topic.iloc[i] == "politics" for i in plain) == 1
    assert sum(live.topic.iloc[i] == "politics" for i in kept) == 3


def test_clicks_are_read_back_for_the_live_demo():
    store = InteractionStore(":memory:")
    for i, (article, feed) in enumerate([("a1", "live"), ("a2", "live"), ("a1", "live"), ("a3", "f1")]):
        store.insert({"event_id": f"e{i}", "user_id": "u", "feed_id": feed, "article_id": article,
                      "event_type": "click", "timestamp": "2026-11-26T10:00:00Z",
                      "displayed_rank": 1, "model_version": "t", "received_at": f"2026-11-26T10:00:0{i}Z"})
    assert store.clicked("u", "live") == ["a1", "a2"]
    assert store.clicked("u") == ["a1", "a2", "a3"]
    assert store.clicked("nobody") == []


def test_todays_reads_are_excluded_and_shift_tastes():
    from agora.api.service import _personal_pool
    from agora.models.live import live_features
    service = _tiny_service()
    builder = service.predictor.builder
    live = pd.DataFrame({"topic": ["sport"] * 3 + ["politics"] * 3, "outlet": "o",
                         "published_at": "2026-11-26T08:00:00Z"})
    vectors = np.array([[1, 0, 0]] * 3 + [[0, 1, 0]] * 3, dtype=float)
    pool = _personal_pool(builder, "u", live, vectors, 3, exclude={0})
    assert 0 not in pool
    before, _ = live_features(builder, "u", live, vectors)
    after, _ = live_features(builder, "u", live, vectors,
                             extra=(vectors[[3, 4]], ["politics", "politics"], ["o", "o"]))
    assert after[3, 1] > before[3, 1]          # topic affinity for politics went up
    assert after[3, 0] > before[3, 0]          # and so did content similarity
