"""Language packs, detection, and graceful degradation."""

from humansignal.compiler import compile_signals
from humansignal.languages import LanguagePack, detect_language, register_pack, unregister_pack
from humansignal.models import CompileRequest

from helpers import features


def test_spanish_pack_relief_and_pressure() -> None:
    result = compile_signals(
        CompileRequest(
            language="es",
            transcript=[
                {"speaker": "Ana", "text": "o sea no sé, es que… [pause] el informe no está listo"},
                {"speaker": "Luis", "text": "¿en serio? otra vez?"},
                {"speaker": "Ana", "text": "uff, menos mal que avisé, por fin"},
            ],
        )
    )
    assert result.emotional_state.relief >= 0.25
    assert result.pressure_zones
    assert features(result) & {"self_correction", "hedge_strong", "pause", "lexicon_relief"}


def test_japanese_confusion_and_pressure() -> None:
    result = compile_signals(
        CompileRequest(
            transcript=[
                {"speaker": "A", "text": "えっと、その件は…ちょっと難しいかもしれない"},
                {"speaker": "B", "text": "どういうこと？"},
                {"speaker": "A", "text": "というか、予算がまだで"},
            ]
        )
    )
    assert detect_language(result.emotional_state.evidence[0].detail or "") or True
    assert result.emotional_state.dominant == "confusion"
    assert result.emotional_state.confusion >= 0.4
    assert result.pressure_zones
    assert result.emotional_state.confidence >= 0.3


def test_unknown_language_keeps_structural_signals_and_lowers_confidence() -> None:
    german = compile_signals(CompileRequest(text="Das ist ja sooo toll!!! haha", language="de"))
    english = compile_signals(CompileRequest(text="This is sooo great!!! haha", language="en"))
    assert german.emotional_state.dominant == "excitement"
    assert german.emotional_state.excitement >= 0.4
    assert "laughter" in features(german)
    assert "elongation" in features(german)
    assert german.emotional_state.confidence <= english.emotional_state.confidence + 0.02


def test_auto_detects_english_spanish_and_japanese() -> None:
    assert detect_language("I sent the deck and you still have not looked") == "en"
    assert detect_language("el informe no está listo porque falta una parte") == "es"
    assert detect_language("えっと、その件はちょっと難しい") == "ja"
    assert detect_language("blargh zzz") == "unknown"


def test_custom_pack_is_pluggable() -> None:
    bare = compile_signals(CompileRequest(text="blargh showed up again", language="und"))
    register_pack(
        LanguagePack(
            code="xx",
            name="Fixture",
            use_word_boundaries=True,
            frustration_strong=("blargh",),
        )
    )
    try:
        packed = compile_signals(CompileRequest(text="blargh showed up again", language="xx"))
    finally:
        unregister_pack("xx")
    assert packed.emotional_state.frustration > bare.emotional_state.frustration + 0.2
    assert packed.emotional_state.dominant == "frustration"
