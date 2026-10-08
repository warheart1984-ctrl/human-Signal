"""Compile a request into schema 1.0.0 JSON.

The pipeline is a pure function: normalize, extract, detect, assemble.
Session memory and the optional enhancer live outside this module so the
default path stays deterministic and offline.
"""

from __future__ import annotations

from humansignal.detectors import (
    choose_style,
    dyad_stats,
    momentum_zones,
    pressure_zones,
    score_connection,
    score_emotions,
    score_intent,
    score_signal_strength,
)
from humansignal.document import build_document
from humansignal.features import Analysis, extract_features
from humansignal.models import CompileRequest, CompileResponse, Segment


def analyze_request(request: CompileRequest, prior: list[Segment] | None = None) -> Analysis:
    """Run normalization and feature extraction. Useful for tests and pack debugging."""
    return extract_features(build_document(request, prior))


def compile_signals(request: CompileRequest, prior: list[Segment] | None = None) -> CompileResponse:
    """Turn text or a transcript into an explained signal reading."""
    analysis = analyze_request(request, prior)
    emotions = score_emotions(analysis)
    dyad = dyad_stats(analysis)
    intent = score_intent(analysis, emotions, dyad)
    pressure = pressure_zones(analysis)
    momentum = momentum_zones(analysis)
    connection = score_connection(analysis, dyad)
    strength = score_signal_strength(analysis, drift=intent.drift)
    style = choose_style(analysis, emotions, intent, connection, pressure, momentum, strength)
    return CompileResponse(
        emotional_state=emotions.state,
        signal_strength=strength,
        intent_alignment=intent,
        pressure_zones=pressure,
        momentum_zones=momentum,
        connection_score=connection,
        recommended_response_style=style,
    )
