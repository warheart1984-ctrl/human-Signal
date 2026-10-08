"""Pressure zones: fillers, pauses, hedges, self-corrections, deflection."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest

from helpers import features


def test_hesitation_cluster_becomes_a_zone() -> None:
    result = compile_signals(
        CompileRequest(language="en", text="um I guess... [pause] maybe we should not")
    )
    assert result.pressure_zones
    assert result.pressure_zones[0].intensity >= 0.38
    zone_features = {item.feature for item in result.pressure_zones[0].evidence}
    assert zone_features & {"filler", "pause", "hedge_strong", "ellipsis"}
    assert result.pressure_zones[0].topic
    assert result.pressure_zones[0].confidence > 0


def test_plain_sentence_has_no_pressure_zone() -> None:
    result = compile_signals(
        CompileRequest(language="en", text="The deploy finished. Logs are in the channel.")
    )
    assert result.pressure_zones == []
    assert "pause" not in features(result)


def test_commitment_ask_is_pressure_without_hesitation() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            text="We test tenant routing with the $5K filter. One paid transaction, one verified route.",
        )
    )
    assert result.pressure_zones
    evidence = {item.feature for zone in result.pressure_zones for item in zone.evidence}
    assert "commitment" in evidence
    span = " ".join(zone.span.text for zone in result.pressure_zones)
    assert "$5K" in span or "paid" in span.lower()
    slogan = compile_signals(CompileRequest(language="en", text="The Ghost doesn't pay rent."))
    assert slogan.pressure_zones == []


def test_currency_amount_works_without_a_language_pack() -> None:
    result = compile_signals(CompileRequest(text="Send $5,000 today.", language="und"))
    assert "commitment" in features(result)
    assert result.pressure_zones
