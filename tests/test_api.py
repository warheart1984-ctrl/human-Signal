"""HTTP contract: health, compile, validation, session memory, enhancer fallback."""

from fastapi.testclient import TestClient

from humansignal.enhancer import HttpEnhancer, NoOpEnhancer, load_enhancer
from humansignal.main import create_app
from humansignal.models import CompileRequest, CompileResponse
from humansignal.session import SessionStore

TOP_LEVEL = {
    "schema_version",
    "emotional_state",
    "signal_strength",
    "intent_alignment",
    "pressure_zones",
    "momentum_zones",
    "connection_score",
    "recommended_response_style",
}


def test_health_and_compile_contract() -> None:
    client = TestClient(create_app(enhancer=NoOpEnhancer(), sessions=SessionStore()))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "schema_version": "1.0.0"}

    response = client.post(
        "/compile",
        json={"language": "en", "text": "You never listen. This is the third time."},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == TOP_LEVEL
    assert body["schema_version"] == "1.0.0"
    assert "X-HumanSignal-Elapsed-Ms" in response.headers
    CompileResponse.model_validate(body)


def test_validation_errors() -> None:
    client = TestClient(create_app(enhancer=NoOpEnhancer(), sessions=SessionStore()))
    both = client.post("/compile", json={"text": "hi", "transcript": [{"text": "hi"}]})
    assert both.status_code == 422
    empty = client.post("/compile", json={})
    assert empty.status_code == 422
    backwards = client.post(
        "/compile",
        json={"transcript": [{"text": "hi", "start": 2, "end": 1}]},
    )
    assert backwards.status_code == 422


def test_session_carries_prior_turns_and_reset_clears_them() -> None:
    client = TestClient(create_app(enhancer=NoOpEnhancer(), sessions=SessionStore()))
    first = client.post(
        "/compile",
        json={
            "session_id": "live-1",
            "language": "en",
            "speaker": "A",
            "text": "You never listen. This is the third time you ignored me.",
        },
    )
    assert first.status_code == 200
    second = client.post(
        "/compile",
        json={"session_id": "live-1", "language": "en", "speaker": "B", "text": "Fine."},
    )
    assert second.status_code == 200
    assert second.json()["intent_alignment"]["drift"] >= 0.55

    reset = client.post(
        "/compile",
        json={
            "session_id": "live-1",
            "reset_session": True,
            "language": "en",
            "speaker": "B",
            "text": "Fine.",
        },
    )
    assert reset.status_code == 200
    assert reset.json()["intent_alignment"]["drift"] < second.json()["intent_alignment"]["drift"]


def test_stateless_context_matches_the_session_story() -> None:
    client = TestClient(create_app(enhancer=NoOpEnhancer(), sessions=SessionStore()))
    response = client.post(
        "/compile",
        json={
            "language": "en",
            "speaker": "B",
            "text": "Fine.",
            "context": [
                {
                    "speaker": "A",
                    "text": "You never listen. This is the third time you ignored me.",
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["intent_alignment"]["label"] == "contradictory"


def test_default_enhancer_is_noop_and_http_failure_keeps_the_draft() -> None:
    assert isinstance(load_enhancer(), NoOpEnhancer)
    draft = create_app(enhancer=NoOpEnhancer(), sessions=SessionStore())
    client = TestClient(draft)
    payload = {"language": "en", "text": "hello there, the build is green"}
    baseline = client.post("/compile", json=payload).json()
    enhancer = HttpEnhancer("http://127.0.0.1:9/enhance", timeout_s=0.2)
    updated = enhancer.enhance(CompileRequest.model_validate(payload), CompileResponse.model_validate(baseline))
    assert updated.model_dump(exclude_none=True) == baseline
