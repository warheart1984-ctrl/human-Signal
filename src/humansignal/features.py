"""Feature extraction.

Structural cues (punctuation, case, repetition, laughter, timing, overlap)
are always on. Lexicon hits are added only when a language pack is active.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from humansignal.document import Document, Turn
from humansignal.languages import LanguagePack, phrase_patterns
from humansignal.scoring import clamp
from humansignal.textutil import (
    ACRONYMS,
    CAPS_RE,
    CORE_FILLER_RE,
    CJK_OR_HANGUL_LAUGH_RE,
    ELLIPSIS_RE,
    ELONGATION_RE,
    EMOJI_RE,
    EMOTICON_NEG_RE,
    EMOTICON_POS_RE,
    EXCLAIM_RE,
    INTERROBANG_RE,
    LAUGHTER_RE,
    MARKER_RE,
    QUESTION_RE,
    SARCASM_SLASH_RE,
    is_emphatic_elongation,
    letter_count,
    normalize_match_text,
    overlaps,
)

LAUGH_MARKERS = frozenset(
    {"laughter", "laugh", "laughs", "laughing", "chuckle", "chuckles", "giggle", "笑"}
)
PAUSE_MARKERS = frozenset({"pause", "silence", "beat"})
LONG_PAUSE_MARKERS = frozenset({"long pause", "long silence"})
OVERLAP_MARKERS = frozenset({"overlap", "crosstalk", "interruption", "interrupt"})
FILLER_MARKERS = frozenset({"filler", "hesitation", "hesitates", "um", "uh"})

# Amounts are structural: "$5K" and "€20" do not need a language pack.
CURRENCY_RE = re.compile(
    r"(?i)(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?\s?[kmb]?|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|dollars|bucks)\b)"
)
# "again" is neutral unless a negation sits just in front of it, or the token is punched up.
_LOADED_REPEAT_PREFIX = re.compile(
    r"(?i)(?:\byet\s+|\b(?:not|never|don'?t|doesn'?t|didn'?t|won'?t|can'?t|isn'?t|aren'?t)(?:\s+\w+){0,4}\s+)$"
)
_LOADED_PUNCT = re.compile(r"^\s*([!！?？]{1,4})")
_FRICTION_FEATURES = frozenset(
    {
        "lexicon_frustration",
        "sarcasm",
        "emoji_negative",
        "emoji_sarcasm",
        "negation",
        "deflection",
        "dismissive",
    }
)

POSITIVE_EMOJI = set("😀😁😄😅😊😍🥰😘❤💕💖💗💙💚💛🧡💜🤗👏🙌👍👌💯🤝🎉")
NEGATIVE_EMOJI = set("😠😡🤬😤😒😞😔😢😭💔👎")
SARCASM_EMOJI = set("🙃😏🙄")
ENERGY_EMOJI = set("🔥💥⚡🚀✨🎉")
LAUGH_EMOJI = set("😂🤣😆")

CORE_NEGATION = frozenset({"no", "not", "nah", "nope"})

CATEGORY_FEATURE = {
    "frustration_strong": "lexicon_frustration",
    "frustration_weak": "lexicon_frustration",
    "excitement_strong": "lexicon_excitement",
    "excitement_weak": "lexicon_excitement",
    "confusion": "lexicon_confusion",
    "relief": "lexicon_relief",
    "positive": "lexicon_positive",
    "agreement": "agreement",
    "hedge_strong": "hedge_strong",
    "hedge_weak": "hedge_weak",
    "filler": "filler",
    "self_correction": "self_correction",
    "deflection": "deflection",
    "repair": "repair",
    "sarcasm": "sarcasm",
    "backchannel": "backchannel",
    "commitment_strong": "commitment",
    "commitment_weak": "commitment",
}

CATEGORY_WEIGHT = {
    "frustration_strong": 1.0,
    "frustration_weak": 0.55,
    "excitement_strong": 1.0,
    "excitement_weak": 0.6,
    "confusion": 0.9,
    "relief": 0.95,
    "positive": 0.7,
    "agreement": 0.65,
    "hedge_strong": 0.8,
    "hedge_weak": 0.45,
    "filler": 0.75,
    "self_correction": 0.8,
    "deflection": 0.85,
    "repair": 0.8,
    "sarcasm": 0.9,
    "backchannel": 0.7,
    # One strong ask clears the pressure floor. A weak word ("pay", "sign") does not.
    "commitment_strong": 0.95,
    "commitment_weak": 0.28,
}

ENERGY_FEATURES = frozenset(
    {
        "exclamation",
        "interrobang",
        "elongation",
        "caps",
        "emoji_positive",
        "emoji_energy",
        "laughter",
    }
)

_DROP_WHEN_DISMISSIVE = frozenset({"lexicon_positive", "agreement", "backchannel"})


@dataclass(frozen=True)
class Hit:
    feature: str
    detail: str
    weight: float
    turn_index: int
    start_char: int | None
    end_char: int | None
    excerpt: str
    speaker: str | None
    start_s: float | None
    end_s: float | None
    role: str
    local_start: int
    local_end: int


@dataclass
class Analysis:
    doc: Document
    hits: list[Hit] = field(default_factory=list)
    by_turn: dict[int, list[Hit]] = field(default_factory=dict)

    @property
    def current_token_count(self) -> int:
        return sum(turn.token_count for turn in self.doc.turns if turn.role == "current")

    @property
    def context_token_count(self) -> int:
        return sum(turn.token_count for turn in self.doc.turns if turn.role == "context")


def extract_features(doc: Document) -> Analysis:
    analysis = Analysis(doc=doc)
    for turn in doc.turns:
        hits = _extract_turn(turn, doc.pack)
        analysis.by_turn[turn.index] = hits
        analysis.hits.extend(hits)
    _resolve_short_replies(doc, analysis)
    _add_timing_features(doc, analysis)
    _add_length_spikes(doc, analysis)
    return analysis


def _global_span(turn: Turn, local_start: int, local_end: int) -> tuple[int | None, int | None]:
    if turn.role != "current" or turn.base_char < 0:
        return None, None
    return turn.base_char + local_start, turn.base_char + local_end


def _hit(
    turn: Turn,
    feature: str,
    detail: str,
    weight: float,
    local_start: int,
    local_end: int,
    excerpt: str,
) -> Hit:
    start_char, end_char = _global_span(turn, local_start, local_end)
    return Hit(
        feature=feature,
        detail=detail,
        weight=weight,
        turn_index=turn.index,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        speaker=turn.speaker,
        start_s=turn.start,
        end_s=turn.end,
        role=turn.role,
        local_start=local_start,
        local_end=local_end,
    )


def _append(bucket: list[Hit], hit: Hit) -> None:
    bucket.append(hit)


def _extract_turn(turn: Turn, pack: LanguagePack | None) -> list[Hit]:
    hits: list[Hit] = []
    text = turn.text
    laughter_spans = _laughter_and_markers(turn, hits)
    _emoji(turn, hits, laughter_spans)
    _punctuation(turn, hits)
    _currency(turn, hits)
    _caps(turn, hits, laughter_spans)
    _elongation(turn, hits, laughter_spans)
    _repetition_and_negation(turn, hits, pack)
    _core_fillers(turn, hits, laughter_spans)
    if SARCASM_SLASH_RE.search(text):
        match = SARCASM_SLASH_RE.search(text)
        assert match is not None
        _append(
            hits,
            _hit(turn, "sarcasm", "explicit /s marker", 0.95, match.start(), match.end(), match.group(0).strip()),
        )
    if pack is not None:
        _pack_hits(turn, pack, hits)
        _leading_self_correction(turn, pack, hits)
        _loaded_repeats(turn, pack, hits)
    return hits


def _laughter_and_markers(turn: Turn, hits: list[Hit]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in LAUGHTER_RE.finditer(turn.text):
        spans.append((match.start(), match.end()))
        _append(
            hits,
            _hit(
                turn,
                "laughter",
                f'laughter "{match.group(0)}"',
                0.9,
                match.start(),
                match.end(),
                match.group(0),
            ),
        )
    for match in MARKER_RE.finditer(turn.text):
        label = re.sub(r"\s+", " ", match.group(1).strip().lower())
        spans.append((match.start(), match.end()))
        if label in LAUGH_MARKERS:
            _append(hits, _hit(turn, "laughter", f"marker [{label}]", 0.95, match.start(), match.end(), match.group(0)))
        elif label in LONG_PAUSE_MARKERS:
            _append(hits, _hit(turn, "pause", f"marker [{label}]", 1.0, match.start(), match.end(), match.group(0)))
        elif label in PAUSE_MARKERS:
            _append(hits, _hit(turn, "pause", f"marker [{label}]", 0.85, match.start(), match.end(), match.group(0)))
        elif label in OVERLAP_MARKERS:
            _append(hits, _hit(turn, "overlap", f"marker [{label}]", 0.8, match.start(), match.end(), match.group(0)))
        elif label in FILLER_MARKERS:
            _append(hits, _hit(turn, "filler", f"marker [{label}]", 0.7, match.start(), match.end(), match.group(0)))
    return spans


def _emoji(turn: Turn, hits: list[Hit], blocked: list[tuple[int, int]]) -> None:
    for match in EMOJI_RE.finditer(turn.text):
        cluster = match.group(0)
        chars = set(cluster)
        if chars & SARCASM_EMOJI:
            feature, detail, weight = "emoji_sarcasm", "sarcasm emoji", 0.85
        elif chars & LAUGH_EMOJI:
            feature, detail, weight = "laughter", "laughter emoji", 0.8
        elif chars & NEGATIVE_EMOJI:
            feature, detail, weight = "emoji_negative", "negative emoji", 0.8
        elif chars & ENERGY_EMOJI:
            feature, detail, weight = "emoji_energy", "energy emoji", 0.7
        elif chars & POSITIVE_EMOJI:
            feature, detail, weight = "emoji_positive", "positive emoji", 0.65
        else:
            feature, detail, weight = "emoji_energy", "emoji", 0.35
        _append(hits, _hit(turn, feature, f'{detail} "{cluster}"', weight, match.start(), match.end(), cluster))
        blocked.append((match.start(), match.end()))
    for pattern, feature, detail in (
        (EMOTICON_POS_RE, "emoji_positive", "positive emoticon"),
        (EMOTICON_NEG_RE, "emoji_negative", "negative emoticon"),
    ):
        for match in pattern.finditer(turn.text):
            if overlaps(match.start(), match.end(), blocked):
                continue
            _append(
                hits,
                _hit(turn, feature, f'{detail} "{match.group(0)}"', 0.45, match.start(), match.end(), match.group(0)),
            )


def _punctuation(turn: Turn, hits: list[Hit]) -> None:
    text = turn.text
    if INTERROBANG_RE.search(text):
        match = INTERROBANG_RE.search(text)
        assert match is not None
        _append(hits, _hit(turn, "interrobang", 'mixed "?!" punctuation', 0.9, match.start(), match.end(), match.group(0)))
        return
    questions = list(QUESTION_RE.finditer(text))
    exclaims = list(EXCLAIM_RE.finditer(text))
    if any(len(match.group(0)) >= 2 for match in questions):
        match = next(match for match in questions if len(match.group(0)) >= 2)
        _append(hits, _hit(turn, "question", "question cluster", 1.0, match.start(), match.end(), match.group(0)))
    elif questions or "¿" in text:
        if questions:
            match = questions[0]
            start, end, excerpt = match.start(), match.end(), match.group(0)
        else:
            start = text.index("¿")
            end = start + 1
            excerpt = "¿"
        _append(hits, _hit(turn, "question", "question mark", 0.62, start, end, excerpt))
    if any(len(match.group(0)) >= 2 for match in exclaims):
        match = next(match for match in exclaims if len(match.group(0)) >= 2)
        _append(hits, _hit(turn, "exclamation", "exclamation cluster", 1.0, match.start(), match.end(), match.group(0)))
    elif exclaims:
        match = exclaims[0]
        _append(hits, _hit(turn, "exclamation", "exclamation mark", 0.7, match.start(), match.end(), match.group(0)))
    ellipsis = ELLIPSIS_RE.search(text)
    if ellipsis:
        _append(
            hits,
            _hit(turn, "ellipsis", "ellipsis", 0.75, ellipsis.start(), ellipsis.end(), ellipsis.group(0)),
        )


def _caps(turn: Turn, hits: list[Hit], blocked: list[tuple[int, int]]) -> None:
    for match in CAPS_RE.finditer(turn.text):
        word = match.group(0)
        if letter_count(word) < 3 or word in ACRONYMS:
            continue
        if overlaps(match.start(), match.end(), blocked):
            continue
        if LAUGHTER_RE.fullmatch(word) or CJK_OR_HANGUL_LAUGH_RE.fullmatch(word):
            continue
        _append(hits, _hit(turn, "caps", f'caps "{word}"', 0.85, match.start(), match.end(), word))


def _elongation(turn: Turn, hits: list[Hit], blocked: list[tuple[int, int]]) -> None:
    seen: set[tuple[int, int]] = set()
    for match in ELONGATION_RE.finditer(turn.text):
        if overlaps(match.start(), match.end(), blocked):
            continue
        left = match.start()
        right = match.end()
        while left > 0 and turn.text[left - 1].isalpha():
            left -= 1
        while right < len(turn.text) and turn.text[right].isalpha():
            right += 1
        if (left, right) in seen:
            continue
        seen.add((left, right))
        word = turn.text[left:right]
        if LAUGHTER_RE.fullmatch(word):
            continue
        if not is_emphatic_elongation(word):
            continue
        _append(hits, _hit(turn, "elongation", f'elongated "{word}"', 0.85, left, right, word))


def _currency(turn: Turn, hits: list[Hit]) -> None:
    for match in CURRENCY_RE.finditer(turn.text):
        surface = match.group(0)
        _append(hits, _hit(turn, "commitment", f'commitment "{surface}"', 0.95, match.start(), match.end(), surface))


def _loaded_repeats(turn: Turn, pack: LanguagePack, hits: list[Hit]) -> None:
    """Count tokens like ``again`` only inside a frustrated construction."""
    text = turn.text
    for token in pack.loaded_repeats:
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(token)}(?!\w)")
        for match in pattern.finditer(text):
            window_start = max(0, match.start() - 48)
            before = text[window_start : match.start()]
            prefix = _LOADED_REPEAT_PREFIX.search(before)
            punct = _LOADED_PUNCT.match(text[match.end() : match.end() + 4])
            emphatic = False
            punct_len = 0
            if punct is not None:
                cluster = punct.group(1)
                emphatic = ("!" in cluster or "！" in cluster or cluster.count("?") + cluster.count("？") >= 2)
                if emphatic:
                    punct_len = len(cluster)
            if prefix is None and not emphatic:
                continue
            raw_start = window_start + prefix.start() if prefix is not None else match.start()
            raw_end = match.end() + punct_len
            excerpt = text[raw_start:raw_end].strip()
            if not excerpt:
                continue
            local_start = raw_start + (len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].lstrip()))
            local_end = local_start + len(excerpt)
            if any(
                hit.feature == "lexicon_frustration"
                and not (local_end <= hit.local_start or local_start >= hit.local_end)
                for hit in hits
            ):
                continue
            _append(
                hits,
                _hit(turn, "lexicon_frustration", f'lexicon frustration "{excerpt}"', 0.9, local_start, local_end, excerpt),
            )


def _repetition_and_negation(turn: Turn, hits: list[Hit], pack: LanguagePack | None) -> None:
    words = turn.words
    negation = set(CORE_NEGATION)
    if pack is not None:
        negation |= {item.lower() for item in pack.negation}
    index = 0
    while index < len(words) - 1:
        current = words[index][0].lower()
        if current in negation and words[index + 1][0].lower() == current:
            start = words[index][1]
            end = words[index + 1][2]
            run = 2
            cursor = index + 2
            while cursor < len(words) and words[cursor][0].lower() == current:
                end = words[cursor][2]
                run += 1
                cursor += 1
            excerpt = turn.text[start:end]
            _append(hits, _hit(turn, "negation", f'repeated negation "{current}" x{run}', 0.9, start, end, excerpt))
            index = cursor
            continue
        index += 1

    for index in range(1, len(words)):
        if words[index][0].lower() != words[index - 1][0].lower():
            continue
        start = words[index - 1][1]
        end = words[index][2]
        if words[index][0].lower() in negation:
            continue
        excerpt = turn.text[start:end]
        _append(
            hits,
            _hit(turn, "repetition", f'repeated "{words[index][0]}"', 0.75, start, end, excerpt),
        )

    for index in range(len(words) - 3):
        left = (words[index][0].lower(), words[index + 1][0].lower())
        right = (words[index + 2][0].lower(), words[index + 3][0].lower())
        if left != right or left[0] == left[1]:
            continue
        start = words[index][1]
        end = words[index + 3][2]
        excerpt = turn.text[start:end]
        _append(hits, _hit(turn, "repetition", f'repeated phrase "{excerpt}"', 0.8, start, end, excerpt))
        break


def _core_fillers(turn: Turn, hits: list[Hit], blocked: list[tuple[int, int]]) -> None:
    for match in CORE_FILLER_RE.finditer(turn.text):
        if overlaps(match.start(), match.end(), blocked):
            continue
        _append(
            hits,
            _hit(turn, "filler", f'filler "{match.group(0)}"', 0.8, match.start(), match.end(), match.group(0)),
        )


def _collect_pack_matches(text: str, pack: LanguagePack) -> list[tuple[int, int, str, str]]:
    found: list[tuple[int, int, str, str]] = []
    if pack.use_word_boundaries:
        lowered = normalize_match_text(text)
        for category, pattern in phrase_patterns(pack):
            for match in pattern.finditer(lowered):
                found.append((match.start(), match.end(), text[match.start() : match.end()], category))
    else:
        for category, phrases in pack.categories().items():
            for phrase in phrases:
                start = 0
                while True:
                    index = text.find(phrase, start)
                    if index < 0:
                        break
                    found.append((index, index + len(phrase), phrase, category))
                    start = index + len(phrase)
    found.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[3]))
    accepted: list[tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, surface, category in found:
        if overlaps(start, end, occupied):
            continue
        occupied.append((start, end))
        accepted.append((start, end, surface, category))
    accepted.sort(key=lambda item: item[0])
    return accepted


def _pack_hits(turn: Turn, pack: LanguagePack, hits: list[Hit]) -> None:
    energy = any(hit.feature in ENERGY_FEATURES for hit in hits)
    for start, end, surface, category in _collect_pack_matches(turn.text, pack):
        if category == "excitement_weak" and not energy:
            continue
        if category == "backchannel" and turn.token_count > 4:
            continue
        if category == "sarcasm" and turn.token_count > 6:
            continue
        feature = CATEGORY_FEATURE[category]
        weight = CATEGORY_WEIGHT[category]
        detail = f'{feature.replace("_", " ")} "{surface}"'
        _append(hits, _hit(turn, feature, detail, weight, start, end, surface))


def _leading_self_correction(turn: Turn, pack: LanguagePack, hits: list[Hit]) -> None:
    if not pack.leading_self_corrections:
        return
    lowered = normalize_match_text(turn.text)
    index = 0
    while index < len(lowered) and lowered[index].isspace():
        index += 1
    if any(hit.local_start == index for hit in hits if hit.feature == "lexicon_confusion"):
        return
    for phrase in sorted(pack.leading_self_corrections, key=len, reverse=True):
        if not lowered.startswith(phrase, index):
            continue
        end = index + len(phrase)
        if end != len(lowered) and lowered[end].isalnum():
            continue
        if any(not (end <= hit.local_start or index >= hit.local_end) for hit in hits):
            return
        _append(
            hits,
            _hit(
                turn,
                "self_correction",
                f'leading repair "{turn.text[index:end]}"',
                0.7,
                index,
                end,
                turn.text[index:end],
            ),
        )
        return


def _short_reply_words(turn: Turn, pack: LanguagePack) -> list[str] | None:
    """Words of a clipped reply made only of dismissive/acknowledgement tokens."""
    if turn.token_count == 0 or turn.token_count > 3 or not pack.dismissives:
        return None
    cleaned = normalize_match_text(turn.text)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"[^\w\s']+", " ", cleaned)
    words = [word for word in cleaned.split() if word not in {"oh", "ya", "ah"}]
    if not words or len(words) > 2:
        return None
    allowed = {item.lower() for item in pack.dismissives}
    if not all(word in allowed for word in words):
        return None
    return words


def _recent_friction(turns: list[Turn], analysis: Analysis, index: int) -> bool:
    """Friction in the last few turns: a clipped reply there is a shutdown, not a nod."""
    start = max(0, index - 3)
    for earlier in turns[start:index]:
        names = {hit.feature for hit in analysis.by_turn.get(earlier.index, [])}
        if names & _FRICTION_FEATURES:
            return True
    return False


def _resolve_short_replies(doc: Document, analysis: Analysis) -> None:
    """``ok`` / ``fine`` are dismissive after recent friction, acknowledgements otherwise.

    Inherent shutdowns (``whatever``, ``どうでも``) stay dismissive with no prior heat.
    Words that also work as agreement or backchannel need frustration, sarcasm,
    a negation burst, or another dismissive in the last three turns.
    """
    pack = doc.pack
    if pack is None:
        return
    ack_pool = {
        item.lower()
        for item in (*pack.backchannels, *pack.agreement, *pack.positive)
    }
    for index, turn in enumerate(doc.turns):
        words = _short_reply_words(turn, pack)
        if words is None:
            continue
        hits = analysis.by_turn[turn.index]
        surface = " ".join(words)
        start = normalize_match_text(turn.text).find(words[0])
        if start < 0:
            start = 0
        end = min(len(turn.text), start + len(surface))
        excerpt = turn.text[start:end] or surface
        ambiguous = all(word in ack_pool for word in words)
        if ambiguous and not _recent_friction(doc.turns, analysis, index):
            hits[:] = [hit for hit in hits if hit.feature != "lexicon_positive"]
            if not any(hit.feature == "backchannel" for hit in hits):
                _append(
                    hits,
                    _hit(turn, "backchannel", f'acknowledgement "{surface}"', 0.7, start, end, excerpt),
                )
            continue
        _append(hits, _hit(turn, "dismissive", f'dismissive "{surface}"', 0.9, start, end, excerpt))
        hits[:] = [hit for hit in hits if hit.feature not in _DROP_WHEN_DISMISSIVE]
    analysis.hits = [hit for turn in doc.turns for hit in analysis.by_turn.get(turn.index, [])]


def _add_timing_features(doc: Document, analysis: Analysis) -> None:
    turns = doc.turns
    for prev, turn in zip(turns, turns[1:]):
        if prev.end is None or turn.start is None:
            continue
        gap = turn.start - prev.end
        if gap >= 0.85:
            weight = clamp((gap - 0.45) / 1.6, 0.4, 1.0)
            excerpt = turn.text.strip()[:32] or "[pause]"
            hit = _hit(turn, "pause", f"pause gap {gap:.2f}s", weight, 0, min(len(turn.text), len(excerpt)), excerpt)
            analysis.by_turn[turn.index].append(hit)
            analysis.hits.append(hit)
        elif turn.start < prev.end - 0.08:
            overlap_s = prev.end - turn.start
            weight = clamp(overlap_s / 0.8, 0.4, 1.0)
            excerpt = turn.text.strip()[:32] or "[overlap]"
            hit = _hit(
                turn,
                "overlap",
                f"overlap {overlap_s:.2f}s",
                weight,
                0,
                min(len(turn.text), len(excerpt)),
                excerpt,
            )
            analysis.by_turn[turn.index].append(hit)
            analysis.hits.append(hit)

    timed: list[tuple[Turn, float]] = []
    for turn in turns:
        if turn.start is None or turn.end is None or turn.end <= turn.start or turn.token_count < 1:
            continue
        timed.append((turn, turn.token_count / (turn.end - turn.start)))
    median = statistics.median([rate for _, rate in timed]) if len(timed) >= 2 else None
    for turn, rate in timed:
        duration = (turn.end or 0) - (turn.start or 0)
        spiked = (median is not None and rate >= max(3.2, 1.45 * median)) or rate >= 4.8
        dropped = (median is not None and rate <= 0.62 * median and turn.token_count >= 4) or (
            rate <= 1.05 and turn.token_count >= 4 and duration >= 1.4
        )
        if spiked:
            detail = f"speaking rate {rate:.1f} w/s"
            if median is not None:
                detail += f" vs median {median:.1f}"
            hit = _hit(turn, "rate_spike", detail, 0.8, 0, min(len(turn.text), 24), turn.text.strip()[:24])
            analysis.by_turn[turn.index].append(hit)
            analysis.hits.append(hit)
        elif dropped:
            detail = f"slow rate {rate:.1f} w/s"
            if median is not None:
                detail += f" vs median {median:.1f}"
            hit = _hit(turn, "rate_drop", detail, 0.7, 0, min(len(turn.text), 24), turn.text.strip()[:24])
            analysis.by_turn[turn.index].append(hit)
            analysis.hits.append(hit)


def _add_length_spikes(doc: Document, analysis: Analysis) -> None:
    counts = [turn.token_count for turn in doc.turns if turn.token_count]
    if len(counts) < 3:
        return
    median = statistics.median(counts)
    for turn in doc.turns:
        if turn.token_count < max(12, 1.8 * median):
            continue
        names = {hit.feature for hit in analysis.by_turn.get(turn.index, [])}
        if not names & ENERGY_FEATURES:
            continue
        hit = _hit(
            turn,
            "length_spike",
            f"long turn {turn.token_count} tokens vs median {median:.0f}",
            0.8,
            0,
            min(len(turn.text), 32),
            turn.text.strip()[:32],
        )
        analysis.by_turn[turn.index].append(hit)
        analysis.hits.append(hit)
