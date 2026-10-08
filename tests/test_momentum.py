"""Momentum zones: rate, length, exclamation, elongation."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, Segment


def test_rate_and_exclamation_spike() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            transcript=[
                Segment(speaker="A", text="ok sure", start=0.0, end=1.2),
                Segment(speaker="A", text="wait", start=1.5, end=2.2),
                Segment(
                    speaker="A",
                    text="YES go go go now now now!!",
                    start=2.3,
                    end=2.8,
                ),
            ],
        )
    )
    assert result.momentum_zones
    assert result.momentum_zones[0].intensity >= 0.4
    names = {item.feature for zone in result.momentum_zones for item in zone.evidence}
    assert names & {"exclamation", "rate_spike", "repetition", "caps", "elongation"}


def test_flat_turn_is_not_momentum() -> None:
    result = compile_signals(
        CompileRequest(language="en", text="The notes are in the shared folder for review.")
    )
    assert result.momentum_zones == []
