"""Shared compile helpers for the detector tests."""

from __future__ import annotations

import json
from pathlib import Path

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, CompileResponse

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCHEMA_PATH = ROOT / "schema" / "compile-response.schema.json"

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


def compile_payload(payload: dict) -> CompileResponse:
    return compile_signals(CompileRequest.model_validate(payload))


def load_example(name: str) -> CompileResponse:
    raw = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    return compile_payload(raw)


def features(result: CompileResponse) -> set[str]:
    found: set[str] = set()
    found.update(item.feature for item in result.emotional_state.evidence)
    found.update(item.feature for item in result.signal_strength.evidence)
    found.update(item.feature for item in result.intent_alignment.evidence)
    found.update(item.feature for item in result.connection_score.evidence)
    for zone in [*result.pressure_zones, *result.momentum_zones]:
        found.update(item.feature for item in zone.evidence)
    return found


def assert_unit_interval(result: CompileResponse) -> None:
    state = result.emotional_state
    for value in (state.frustration, state.excitement, state.confusion, state.relief, state.confidence):
        assert 0.0 <= value <= 1.0
    assert 0.0 <= result.signal_strength.score <= 1.0
    assert 0.0 <= result.intent_alignment.score <= 1.0
    assert 0.0 <= result.intent_alignment.drift <= 1.0
    assert 0.0 <= result.connection_score.score <= 1.0
    for zone in [*result.pressure_zones, *result.momentum_zones]:
        assert 0.0 <= zone.intensity <= 1.0
        assert 0.0 <= zone.confidence <= 1.0
