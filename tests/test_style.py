"""Every recommended style is reachable."""

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, Segment


def _style(payload: dict) -> str:
    return compile_signals(CompileRequest.model_validate(payload)).recommended_response_style.style


def test_all_five_styles() -> None:
    assert _style({"language": "en", "text": "You never listen. I am done. This is the third time."}) == "calm"
    assert _style({"language": "en", "text": "The deploy finished. Logs are in the channel."}) == "direct"
    assert (
        _style(
            {
                "language": "en",
                "text": "um wait, I mean... what do you mean, should we return or throw??",
            }
        )
        == "clarifying"
    )
    playful = compile_signals(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="haha you did not"),
                Segment(speaker="B", text="jaja I absolutely did"),
                Segment(speaker="A", text="hahaha okay okay"),
            ]
        )
    )
    assert playful.recommended_response_style.style == "playful"
    hype = compile_signals(
        CompileRequest(language="en", text="YESSS let's go!! this is amazing and I love this!!!")
    )
    assert hype.emotional_state.dominant == "excitement"
    assert hype.recommended_response_style.style == "hype"
