"""Intent drift: clipped agreement, hedges, sarcasm, and consistent wording."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, Segment

from helpers import features


def test_fine_after_frustration_is_contradictory() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            speaker="B",
            text="Fine.",
            context=[
                Segment(speaker="A", text="You never listen. This is the third time you ignored me."),
            ],
        )
    )
    assert result.intent_alignment.drift >= 0.6
    assert result.intent_alignment.label == "contradictory"
    assert result.intent_alignment.score <= 0.45
    assert result.recommended_response_style.style == "calm"
    blob = " ".join(item.detail.lower() for item in result.intent_alignment.evidence)
    assert "fine" in blob or "clipped" in blob or "dismissive" in features(result)


def test_hedged_positive_drifts() -> None:
    result = compile_signals(CompileRequest(language="en", text="Yeah I guess it's fine."))
    assert result.intent_alignment.drift >= 0.34
    assert result.intent_alignment.label in {"drifting", "contradictory"}
    assert "hedged_positive" in features(result)


def test_sarcasm_emoji_drifts() -> None:
    result = compile_signals(CompileRequest(language="en", text="Great job \ud83d\ude43"))
    assert result.intent_alignment.drift >= 0.45
    assert result.recommended_response_style.style == "calm"


def test_sincere_frustration_stays_aligned() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            text="You never listen and this is the third time. I am done explaining it.",
        )
    )
    assert result.intent_alignment.drift < 0.34
    assert result.intent_alignment.label == "aligned"
    assert result.emotional_state.dominant == "frustration"


def test_low_alignment_agreement() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            speaker="B",
            text="agreed",
            context=[
                Segment(
                    speaker="A",
                    text="You never listen and this is the third time you have ignored the deck completely.",
                )
            ],
        )
    )
    assert result.intent_alignment.drift >= 0.34
