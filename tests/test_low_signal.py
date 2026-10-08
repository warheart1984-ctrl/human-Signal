"""Low-signal input stays uncertain instead of inventing a reading."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest

from helpers import assert_unit_interval


def test_short_inputs_are_neutral_and_low_confidence() -> None:
    for text in ("ok", "hi", "thanks"):
        result = compile_signals(CompileRequest(text=text, language="en"))
        assert_unit_interval(result)
        assert result.emotional_state.dominant == "neutral"
        assert result.emotional_state.confidence <= 0.3
        assert result.signal_strength.score <= 0.25
        assert result.signal_strength.confidence <= 0.35
        assert result.intent_alignment.label == "uncertain"
        assert result.pressure_zones == []
        assert result.momentum_zones == []
        assert result.recommended_response_style.style == "direct"
        assert result.recommended_response_style.confidence <= 0.4


def test_same_input_is_deterministic() -> None:
    request = CompileRequest(language="en", text="You never listen. Fine.")
    assert compile_signals(request).model_dump() == compile_signals(request).model_dump()
