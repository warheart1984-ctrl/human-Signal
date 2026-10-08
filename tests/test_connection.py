"""Connection: shared laughter, mirroring, single-speaker humility."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, Segment

from helpers import features


def test_shared_laughter_lifts_connection() -> None:
    result = compile_signals(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="haha you did not"),
                Segment(speaker="B", text="jaja I absolutely did"),
                Segment(speaker="A", text="hahaha okay okay"),
            ]
        )
    )
    assert result.connection_score.components.laughter >= 0.55
    assert result.connection_score.score >= 0.45
    assert result.recommended_response_style.style == "playful"


def test_mirroring_between_speakers() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            transcript=[
                Segment(speaker="A", text="the token bucket resets every minute under load"),
                Segment(
                    speaker="B",
                    text="if the token bucket resets every minute then the burst is the bug",
                ),
            ],
        )
    )
    assert result.connection_score.components.mirroring >= 0.4


def test_ok_after_a_suggestion_is_backchannel() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            transcript=[
                Segment(speaker="Daniel", text="you need to stop building and start watching because it will show you where we are at"),
                Segment(speaker="Jon", text="ok"),
            ],
        )
    )
    assert "dismissive" not in features(result)
    assert "backchannel" in features(result)
    assert result.connection_score.components.backchannel > 0
    ack = " ".join(item.detail for item in result.connection_score.evidence)
    assert "acknowledgement" in ack


def test_whatever_is_dismissive_without_prior_heat() -> None:
    result = compile_signals(CompileRequest(language="en", text="whatever"))
    assert "dismissive" in features(result)
    assert "backchannel" not in features(result)


def test_ok_after_friction_stays_dismissive() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            speaker="Jon",
            text="ok",
            context=[Segment(speaker="Daniel", text="You never listen. This is the third time you ignored me.")],
        )
    )
    assert "dismissive" in features(result)
    assert result.intent_alignment.label in {"drifting", "contradictory"}
    assert result.intent_alignment.drift >= 0.34


def test_single_speaker_connection_stays_modest() -> None:
    result = compile_signals(CompileRequest(language="en", text="haha that was a lot"))
    assert result.connection_score.score < 0.55
    assert result.connection_score.confidence < 0.5
