"""Normalize a compile request into ordered turns.

Context turns (session memory and explicit ``context``) come first. The
current text or transcript follows. Character offsets in the response refer
to ``current_text`` only, which is the current turns joined by newlines.
"""

from __future__ import annotations

from dataclasses import dataclass

from humansignal.errors import InputError
from humansignal.languages import LanguagePack, resolve_language
from humansignal.models import CompileRequest, Segment
from humansignal.textutil import tokenize, word_tokens


@dataclass
class Turn:
    speaker: str | None
    text: str
    start: float | None
    end: float | None
    index: int
    role: str
    base_char: int
    token_count: int
    words: tuple[tuple[str, int, int], ...]


@dataclass
class Document:
    turns: list[Turn]
    current_text: str
    language: str
    pack: LanguagePack | None

    def current_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "current"]

    def context_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "context"]


def _turn_from_segment(segment: Segment, index: int, role: str, base_char: int) -> Turn:
    words = word_tokens(tokenize(segment.text))
    return Turn(
        speaker=segment.speaker,
        text=segment.text,
        start=segment.start,
        end=segment.end,
        index=index,
        role=role,
        base_char=base_char,
        token_count=len(words),
        words=tuple((tok.text, tok.start, tok.end) for tok in words),
    )


def current_segments(request: CompileRequest) -> list[Segment]:
    """The turns this request adds, ignoring context and session memory."""
    if request.text is not None and request.text.strip():
        return [Segment(text=request.text, speaker=request.speaker)]
    return [segment for segment in (request.transcript or []) if segment.text.strip()]


def build_document(request: CompileRequest, prior: list[Segment] | None = None) -> Document:
    context: list[Segment] = [segment for segment in (prior or []) if segment.text.strip()]
    if request.context:
        context.extend(segment for segment in request.context if segment.text.strip())
    current = current_segments(request)
    if not current:
        raise InputError("provide text or transcript")

    sample = "\n".join(segment.text for segment in [*context, *current])
    language, pack = resolve_language(request.language, sample)

    turns: list[Turn] = []
    for segment in context:
        turns.append(_turn_from_segment(segment, len(turns), "context", -1))

    chunks: list[str] = []
    cursor = 0
    for segment in current:
        if chunks:
            chunks.append("\n")
            cursor += 1
        turns.append(_turn_from_segment(segment, len(turns), "current", cursor))
        chunks.append(segment.text)
        cursor += len(segment.text)

    return Document(
        turns=turns,
        current_text="".join(chunks),
        language=language,
        pack=pack,
    )
