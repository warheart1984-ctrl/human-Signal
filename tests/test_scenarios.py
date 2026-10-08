"""The four scenarios called out in the README."""

from helpers import assert_unit_interval, features, load_example


def test_work_conflict() -> None:
    result = load_example("work_conflict.json")
    assert_unit_interval(result)
    assert result.emotional_state.dominant == "frustration"
    assert result.emotional_state.frustration >= 0.55
    assert result.intent_alignment.label in {"drifting", "contradictory"}
    assert result.intent_alignment.drift >= 0.55
    assert result.pressure_zones
    assert result.connection_score.score < 0.45
    assert result.recommended_response_style.style == "calm"
    assert result.emotional_state.confidence >= 0.35


def test_creative_jam() -> None:
    result = load_example("creative_jam.json")
    assert result.emotional_state.dominant == "excitement"
    assert result.emotional_state.excitement >= 0.5
    assert result.connection_score.score >= 0.4
    assert result.momentum_zones
    assert result.recommended_response_style.style in {"hype", "playful"}
    assert "laughter" in features(result)


def test_pair_coding() -> None:
    result = load_example("pair_coding.json")
    assert result.emotional_state.dominant == "confusion"
    assert result.emotional_state.confusion >= 0.4
    assert result.pressure_zones
    assert result.recommended_response_style.style == "clarifying"
    assert features(result) & {"pause", "filler", "self_correction", "question"}


def test_family() -> None:
    result = load_example("family.json")
    assert result.emotional_state.dominant == "frustration"
    assert result.emotional_state.frustration >= 0.4
    assert result.emotional_state.relief < result.emotional_state.frustration
    assert result.recommended_response_style.style == "calm"
    assert "repair" in features(result)
