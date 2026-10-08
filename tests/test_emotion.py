"""Emotional load: frustration, excitement, confusion, relief."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest

from helpers import assert_unit_interval, features


def test_frustration_from_lexicon_and_heat() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            text="You never listen. This is the third time and I am done.",
        )
    )
    assert_unit_interval(result)
    assert result.emotional_state.dominant == "frustration"
    assert result.emotional_state.frustration >= 0.55
    assert result.emotional_state.frustration > result.emotional_state.excitement
    assert "lexicon_frustration" in features(result)


def test_excitement_from_energy_without_a_lexicon() -> None:
    result = compile_signals(CompileRequest(text="Das ist ja sooo toll!!! haha", language="de"))
    assert result.emotional_state.excitement >= 0.45
    assert result.emotional_state.excitement > result.emotional_state.frustration
    assert result.emotional_state.dominant == "excitement"


def test_confusion_from_questions_repairs_and_fillers() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            text="um wait, I mean... what do you mean by that??",
        )
    )
    assert result.emotional_state.dominant == "confusion"
    assert result.emotional_state.confusion >= 0.5
    assert result.emotional_state.confidence >= 0.35


def test_relief_after_load_and_from_lexicon() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            transcript=[
                {"speaker": "A", "text": "You never sent it. I am done with this."},
                {"speaker": "B", "text": "phew, finally. I'm sorry, it's sent."},
            ],
        )
    )
    assert result.emotional_state.relief >= 0.35
    assert "lexicon_relief" in features(result) or "relief_after_load" in features(result)
