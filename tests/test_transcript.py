"""Timestamps change pauses, rate, and overlap."""

from humansignal.compiler import analyze_request, compile_signals
from humansignal.models import CompileRequest, Segment


def test_gap_adds_a_real_pause() -> None:
    tight = analyze_request(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="I sent the deck yesterday.", start=0.0, end=2.0),
                Segment(speaker="B", text="I was in meetings.", start=2.2, end=3.4),
            ]
        )
    )
    gapped = analyze_request(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="I sent the deck yesterday.", start=0.0, end=2.0),
                Segment(speaker="B", text="I was in meetings.", start=4.6, end=7.2),
            ]
        )
    )
    assert not any(hit.feature == "pause" and "gap" in hit.detail for hit in tight.hits)
    pauses = [hit for hit in gapped.hits if hit.feature == "pause" and "gap" in hit.detail]
    assert pauses
    assert "2.60s" in pauses[0].detail or "2.6" in pauses[0].detail


def test_fast_speech_flags_a_rate_spike() -> None:
    analysis = analyze_request(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="we should look at the logs", start=0.0, end=2.5),
                Segment(
                    speaker="B",
                    text="go go open it now and ship the patch before the review starts",
                    start=2.6,
                    end=3.2,
                ),
            ]
        )
    )
    assert any(hit.feature == "rate_spike" for hit in analysis.hits)


def test_overlap_is_visible() -> None:
    result = compile_signals(
        CompileRequest(
            language="en",
            transcript=[
                Segment(speaker="A", text="I was still talking about the deck", start=0.0, end=3.0),
                Segment(speaker="B", text="no listen", start=2.2, end=3.4),
            ],
        )
    )
    assert any(item.feature == "overlap" for item in result.signal_strength.evidence) or any(
        hit.feature == "overlap" for hit in analyze_request(
            CompileRequest(
                transcript=[
                    Segment(speaker="A", text="I was still talking about the deck", start=0.0, end=3.0),
                    Segment(speaker="B", text="no listen", start=2.2, end=3.4),
                ]
            )
        ).hits
    )
