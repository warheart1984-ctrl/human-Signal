"""Regression on the Daniel/Jon transcript in examples/real."""

from helpers import assert_unit_interval, features, load_example


def _details(result) -> str:
    chunks: list[str] = []
    chunks.extend(item.detail for item in result.emotional_state.evidence)
    chunks.extend(item.detail for item in result.signal_strength.evidence)
    chunks.extend(item.detail for item in result.intent_alignment.evidence)
    chunks.extend(item.detail for item in result.connection_score.evidence)
    for zone in [*result.pressure_zones, *result.momentum_zones]:
        chunks.append(zone.span.text)
        chunks.extend(item.detail for item in zone.evidence)
    return " ".join(chunks).lower()


def test_full_transcript_corrects_the_observed_misreads() -> None:
    result = load_example("real/full_transcript.request.json")
    assert_unit_interval(result)
    blob = _details(result)
    assert 'frustration "again"' not in blob
    assert "telll" not in blob
    assert "elongat" not in blob
    assert "dismissive" not in features(result)
    assert result.connection_score.components.backchannel > 0
    assert result.pressure_zones
    pressure = " ".join(zone.span.text for zone in result.pressure_zones)
    assert "$5K" in pressure or "paid transaction" in pressure.lower()
    assert any(item.feature == "commitment" for zone in result.pressure_zones for item in zone.evidence)
    # The ask is evidenced. A one-word acknowledgement is not turned into heat.
    assert result.emotional_state.dominant == "neutral"
    assert result.intent_alignment.drift < 0.34


def test_last_turn_keeps_ok_as_context_acknowledgement() -> None:
    result = load_example("real/jon_last_turn.request.json")
    assert_unit_interval(result)
    blob = _details(result)
    assert "telll" not in blob
    assert 'frustration "again"' not in blob
    assert "dismissive" not in blob
    assert result.connection_score.components.backchannel > 0
    assert result.emotional_state.dominant == "neutral"
    assert result.pressure_zones == []
