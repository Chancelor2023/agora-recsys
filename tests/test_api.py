
from fastapi.testclient import TestClient

from agora.api.main import app


client = TestClient(app)


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
