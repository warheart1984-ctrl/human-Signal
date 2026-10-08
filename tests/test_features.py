"""Structural feature extraction: punctuation, laughter, caps, timing tokens."""

from humansignal.compiler import analyze_request
from humansignal.models import CompileRequest, Segment


def _features(text: str, **kwargs: object) -> set[str]:
    analysis = analyze_request(CompileRequest.model_validate({"text": text, **kwargs}))
    return {hit.feature for hit in analysis.hits}


def test_punctuation_caps_elongation_and_repetition() -> None:
    features = _features("EARLY sooo no no no!!", language="en")
    assert "caps" in features
    assert "elongation" in features
    assert "exclamation" in features
    assert "negation" in features


def test_caps_skip_acronyms_and_laughter_is_not_elongation() -> None:
    analysis = analyze_request(CompileRequest(text="The API is up. hahaha", language="en"))
    features = {hit.feature for hit in analysis.hits}
    assert "caps" not in features
    assert "laughter" in features
    assert "elongation" not in features


def test_laughter_tokens_are_language_independent() -> None:
    for token in ("haha", "jajaja", "www", "555", "哈哈", "mdr", "kkk", "lol", "jeje", "ptdr", "ㅋㅋ"):
        features = _features(f"hey {token}", language="und")
        assert "laughter" in features, token


def test_url_and_digits_are_not_laughter() -> None:
    features = _features("see https://www.example.com and id 55512", language="en")
    assert "laughter" not in features


def test_transcript_markers() -> None:
    analysis = analyze_request(
        CompileRequest(
            transcript=[
                Segment(speaker="A", text="I was [pause] not sure [laughter]", start=0, end=2),
            ]
        )
    )
    features = {hit.feature for hit in analysis.hits}
    assert "pause" in features
    assert "laughter" in features


def test_fillers_and_hedges() -> None:
    features = _features("wait, um I guess we should", language="en")
    assert "filler" in features
    assert "hedge_strong" in features
    assert "self_correction" in features


def test_one_extra_letter_is_not_elongation() -> None:
    assert "elongation" not in _features("telll me what you mean", language="en")
    assert "elongation" not in _features("helllo there", language="en")
    assert "elongation" in _features("sooo yes", language="en")
    assert "elongation" in _features("YESSS", language="en")
    assert "elongation" in _features("hellooo", language="en")
    assert "elongation" in _features("nooo", language="en")


def test_again_is_frustration_only_in_a_loaded_frame() -> None:
    assert "lexicon_frustration" not in _features("I have to look at it again", language="en")
    assert "lexicon_frustration" not in _features("say that again?", language="en")
    for text in ("not again", "yet again", "again?!", "I am not doing this again."):
        assert "lexicon_frustration" in _features(text, language="en"), text
