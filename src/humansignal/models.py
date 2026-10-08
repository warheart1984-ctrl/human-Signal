"""Request and response models. The JSON contract is versioned by ``schema_version``."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from humansignal.version import SCHEMA_VERSION

DominantLabel = Literal["frustration", "excitement", "confusion", "relief", "neutral"]
IntentLabel = Literal["aligned", "drifting", "contradictory", "uncertain"]
ResponseStyle = Literal["calm", "hype", "direct", "playful", "clarifying"]
SpanSource = Literal["current", "context"]


class Segment(BaseModel):
    """One conversational turn, optionally timed and attributed."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)
    speaker: str | None = Field(default=None, max_length=80)
    start: float | None = Field(default=None, ge=0, description="Start time in seconds.")
    end: float | None = Field(default=None, ge=0, description="End time in seconds.")

    @field_validator("start", "end")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("timestamps must be finite")
        return value

    @model_validator(mode="after")
    def _order(self) -> Segment:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class CompileRequest(BaseModel):
    """Input to ``POST /compile``.

    Provide ``text`` or ``transcript``, not both. ``context`` is prior turns
    for stateless drift and momentum. ``session_id`` is the stateful equivalent
    on a single process.
    """

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, max_length=20_000)
    transcript: list[Segment] | None = Field(default=None, max_length=500)
    context: list[Segment] | None = Field(default=None, max_length=500)
    speaker: str | None = Field(
        default=None,
        max_length=80,
        description="Speaker label used when ``text`` is a single turn.",
    )
    language: str | None = Field(
        default=None,
        max_length=16,
        description="BCP-47-ish code (en, es, ja). Omit to detect. Unknown codes use core features only.",
    )
    session_id: str | None = Field(default=None, max_length=128)
    reset_session: bool = False

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        code = value.strip()
        if not code:
            return None
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        if any(ch not in allowed for ch in code):
            raise ValueError("language must be an alphanumeric code")
        return code

    @model_validator(mode="after")
    def _input_shape(self) -> CompileRequest:
        has_text = self.text is not None and self.text.strip() != ""
        has_transcript = bool(self.transcript)
        if has_text and has_transcript:
            raise ValueError("provide text or transcript, not both")
        if not has_text and not has_transcript:
            raise ValueError("provide text or transcript")
        total = len(self.text or "")
        for segment in self.transcript or []:
            total += len(segment.text)
        for segment in self.context or []:
            total += len(segment.text)
        if total > 50_000:
            raise ValueError("input exceeds 50000 characters")
        return self


class Span(BaseModel):
    """A slice of the input that justified an inference."""

    model_config = ConfigDict(extra="forbid")

    text: str
    start_char: int | None = None
    end_char: int | None = None
    speaker: str | None = None
    start: float | None = Field(default=None, description="Turn start in seconds, if provided.")
    end: float | None = Field(default=None, description="Turn end in seconds, if provided.")
    source: SpanSource = "current"


class Evidence(BaseModel):
    """One observable cue. ``weight`` is relative strength, not a probability."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    detail: str
    weight: float = Field(ge=0, le=1)
    span: Span | None = None


class EmotionalState(BaseModel):
    """Per-emotion load in ``[0, 1]``. Emotions are not a partition; several can be high."""

    model_config = ConfigDict(extra="forbid")

    frustration: float = Field(ge=0, le=1)
    excitement: float = Field(ge=0, le=1)
    confusion: float = Field(ge=0, le=1)
    relief: float = Field(ge=0, le=1)
    dominant: DominantLabel
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]


class SignalStrength(BaseModel):
    """How much paralinguistic material the input actually contains."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]


class IntentAlignment(BaseModel):
    """Whether the wording agrees with the other signals.

    ``score`` near 1 means the words and the cues point the same way.
    ``drift`` near 1 means they diverge (flat agreement, sarcasm, clipped "fine.").
    This is not a judgment that someone is lying.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    drift: float = Field(ge=0, le=1)
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]


class Zone(BaseModel):
    """A topic span where pressure or momentum clusters."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    span: Span
    intensity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]


class ConnectionComponents(BaseModel):
    """Pieces of ``connection_score``. Each is 0..1 before they are combined."""

    model_config = ConfigDict(extra="forbid")

    laughter: float = Field(ge=0, le=1)
    mirroring: float = Field(ge=0, le=1)
    rhythm: float = Field(ge=0, le=1)
    backchannel: float = Field(ge=0, le=1)


class ConnectionScore(BaseModel):
    """Dyadic warmth: shared laughter, echo, rhythm. Weak when only one speaker is visible."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    components: ConnectionComponents
    evidence: list[Evidence]


class RecommendedResponseStyle(BaseModel):
    """A stance suggestion, not a script. ``style`` is the enum the caller should branch on."""

    model_config = ConfigDict(extra="forbid")

    style: ResponseStyle
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]


class CompileResponse(BaseModel):
    """Schema 1.0.0 compile result. Top-level keys are part of the contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    emotional_state: EmotionalState
    signal_strength: SignalStrength
    intent_alignment: IntentAlignment
    pressure_zones: list[Zone]
    momentum_zones: list[Zone]
    connection_score: ConnectionScore
    recommended_response_style: RecommendedResponseStyle


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    schema_version: Literal["1.0.0"]
