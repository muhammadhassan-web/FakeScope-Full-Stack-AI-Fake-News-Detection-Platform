"""Unit tests for FakeScope."""

from __future__ import annotations

import pytest

from app import create_app
from app.services.validation import validate_news_text
from preprocess import clean_text


@pytest.fixture()
def app():
    application = create_app("testing")
    application.config.update(TESTING=True)
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_clean_text_strips_urls_and_noise():
    raw = "BREAKING!! Visit https://evil.example.com NOW!!!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "breaking" in cleaned or "visit" in cleaned


def test_validate_news_text_rejects_short():
    text, err = validate_news_text("too short", min_length=20, max_length=100)
    assert text is None
    assert err is not None
    assert "short" in err.message.lower()


def test_validate_news_text_accepts_ok():
    sample = "Federal Reserve raises interest rates amid inflation concerns today."
    text, err = validate_news_text(sample, min_length=20, max_length=1000)
    assert err is None
    assert text == sample


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"FakeScope" in res.data
    assert b"Analyze" in res.data


def test_docs_page(client):
    res = client.get("/docs")
    assert res.status_code == 200
    assert b"/predict" in res.data


def test_about_page(client):
    res = client.get("/about")
    assert res.status_code == 200


def test_predict_requires_json(client):
    res = client.post("/predict", data="not-json", content_type="text/plain")
    assert res.status_code == 400


def test_predict_validates_empty(client):
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 400
    assert res.get_json()["code"] == "validation_error"


def test_health_endpoint(client, app):
    res = client.get("/health")
    # 200 if models present, 503 otherwise — both are valid responses
    assert res.status_code in (200, 503)
    body = res.get_json()
    assert "status" in body
    assert "ready" in body


def test_predict_when_model_ready(client, app):
    from app.services import predictor as predictor_module

    if predictor_module.predictor is None or not predictor_module.predictor.ready:
        pytest.skip("Model artifacts not available")

    payload = {
        "text": (
            "The Federal Reserve increased benchmark rates for the third "
            "consecutive quarter, citing persistent consumer price inflation."
        )
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["label"] in ("FAKE", "REAL")
    assert "confidence" in data
    assert "explanation" in data
