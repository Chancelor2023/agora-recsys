import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import agora.freeze as freeze
from agora.experiment import _finish_test_run, _start_test_run
from agora.models.embeddings import encode_articles, load_text_encoder


@pytest.fixture
def frozen_root(tmp_path, monkeypatch):
    """A fake project with every frozen file, then a manifest."""
    monkeypatch.chdir(tmp_path)
    for path in freeze.frozen_files(**freeze.DEFAULTS).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {path}", encoding="utf-8")
    (tmp_path / "configs/reranking_agents.json").write_text(json.dumps(
        {"lambda": 0.25, "agents": [{"name": "public_affairs", "attribute": "topic",
                                     "protected_values": ["politics"], "target_share": 0.082}]}))
    freeze.main(["create", "--note", "validation run of 20 Nov", "--idea-dir", "missing"])
    return tmp_path


def test_intact_freeze_verifies(frozen_root):
    assert freeze.verify() == []
    manifest = json.loads(freeze.manifest_path("artifacts/models").read_text())
    assert manifest["lambda_test"] == [0.0, 0.25]
    assert manifest["test_rows"][-1] == "Neural + SCRUF-D"


@pytest.mark.parametrize("name", ["artifacts/models/embeddings.npz",
                                  "artifacts/models/embeddings.npz.encoder.joblib",
                                  "data/processed/reads_train.csv",
                                  "data/processed/train_labels.json",
                                  "configs/experiment.toml"])
def test_changed_input_is_detected(frozen_root, name):
    (frozen_root / name).write_text("modified after the freeze")
    assert any(name.split('/')[-1] in p for p in freeze.verify())


def test_changed_lambda_or_target_is_detected(frozen_root):
    path = frozen_root / "configs/reranking_agents.json"
    config = json.loads(path.read_text())
    config["lambda"] = 0.1
    path.write_text(json.dumps(config))
    assert freeze.verify()


def test_changed_code_is_detected(frozen_root, monkeypatch):
    monkeypatch.setattr(freeze, "source_hash", lambda root=None: "edited")
    assert any("code changed" in p for p in freeze.verify())


def test_removed_or_added_optional_file_is_detected(frozen_root):
    (frozen_root / "data/processed/eligible_users.json").unlink()
    assert any("eligible_users.json: missing" in p for p in freeze.verify())


def test_manifest_cannot_be_replaced_after_a_test_run(frozen_root):
    freeze.run_log_path("artifacts/models").write_text('{"status": "started"}\n')
    with pytest.raises(SystemExit):
        freeze.main(["create", "--note", "again", "--replace", "--idea-dir", "missing"])


def _args(tmp_path, **kw):
    return argparse.Namespace(lambdas=kw.get("lambdas"), out=str(tmp_path / "reports"),
                              rerun_reason=kw.get("rerun_reason"), **freeze.DEFAULTS)


def test_test_run_refuses_lambdas_and_unfrozen_state(frozen_root):
    with pytest.raises(SystemExit):
        _start_test_run(_args(frozen_root, lambdas="0,0.1,0.25"))
    (frozen_root / "data/processed/train_labels.json").write_text("changed")
    with pytest.raises(SystemExit):
        _start_test_run(_args(frozen_root))


def test_second_test_run_never_overwrites_the_first(frozen_root):
    first = _start_test_run(_args(frozen_root))
    assert first["stem"] == "experiment_test"
    result = frozen_root / "reports" / "experiment_test.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}")
    _finish_test_run(first, result)
    with pytest.raises(SystemExit):
        _start_test_run(_args(frozen_root))
    second = _start_test_run(_args(frozen_root, rerun_reason="evaluator bug #12"))
    assert second["stem"] == "experiment_test_run2"
    log = [json.loads(l) for l in freeze.run_log_path("artifacts/models").read_text().splitlines()]
    assert [r["status"] for r in log] == ["started", "finished", "started"]


def test_encoder_is_fitted_on_past_articles_only(tmp_path):
    rng = np.random.default_rng(0)
    words = [f"word{i}" for i in range(60)]
    rows = []
    for i in range(160):
        text = " ".join(rng.choice(words, 12))
        rows.append({"article_id": f"a{i:03d}", "title": text, "lead": text, "topic": "politics",
                     "published_at": "2023-11-20T08:00:00Z"})
    # two future articles share a word that never appears before the cut-off
    for i in (0, 1):
        rows.append({"article_id": f"f{i}", "title": "zeitgeistword zeitgeistword", "lead": "future",
                     "topic": "politics", "published_at": "2023-12-05T08:00:00Z"})
    articles = pd.DataFrame(rows)
    out = tmp_path / "emb.npz"
    meta = encode_articles(articles, out, "tfidf", fit_before="2023-11-30T00:00:00+00:00")
    assert meta["fit_articles"] == 160
    tfidf, _ = __import__("joblib").load(str(out) + ".encoder.joblib")
    assert "zeitgeistword" not in tfidf.vocabulary_
    encode = load_text_encoder(out)
    stored = np.load(out)["vectors"]
    np.testing.assert_allclose(encode(["word1 word2\nword3"]).shape[1], stored.shape[1])
