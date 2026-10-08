"""Detectors.

Each one turns features into a scored, evidenced inference. Nothing here
claims to know why a person feels something; it reports which cues fired.

Blend for a side of the conversation (current or context):

    0.55 * token-weighted average + 0.45 * peak

A short current reply (under 6 tokens) inherits prior load, because a clipped
"fine." is not an emotional reset. Zones are only emitted for the current input.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from humansignal.document import Turn
from humansignal.features import Analysis, Hit
from humansignal.models import (
    ConnectionComponents,
    ConnectionScore,
    EmotionalState,
    Evidence,
    IntentAlignment,
    RecommendedResponseStyle,
    ResponseStyle,
    SignalStrength,
    Span,
    Zone,
)
from humansignal.scoring import clamp, squash, unit
from humansignal.textutil import (
    BASIC_STOPWORDS,
    QUOTE_RE,
    char_bigrams,
    cjk_ratio,
    content_words,
    jaccard,
    split_clauses,
    topic_phrase,
)

EMOTIONS = ("frustration", "excitement", "confusion", "relief")

DOMINANT_FLOOR = 0.28
PRESSURE_FLOOR = 0.38
MOMENTUM_FLOOR = 0.40
DRIFT_CONTRADICTORY = 0.62
DRIFT_DRIFTING = 0.34

FEATURE_CAP = {
    "lexicon_frustration": 1.6,
    "lexicon_excitement": 1.6,
    "lexicon_confusion": 1.4,
    "lexicon_relief": 1.4,
    "question": 1.3,
    "exclamation": 1.2,
    "filler": 1.4,
    "caps": 1.5,
    "laughter": 1.4,
    "repetition": 1.2,
    "ellipsis": 0.9,
    "hedge_strong": 1.2,
    "hedge_weak": 1.0,
    "self_correction": 1.3,
    "pause": 1.3,
    "elongation": 1.2,
    "interrobang": 1.1,
}

PRESSURE_COEF = {
    "filler": 0.52,
    "pause": 0.62,
    "hedge_strong": 0.46,
    "hedge_weak": 0.20,
    "self_correction": 0.50,
    "deflection": 0.55,
    "dismissive": 0.40,
    "ellipsis": 0.40,
    "rate_drop": 0.36,
    "overlap": 0.28,
    # A money or commitment ask is pressure even when nobody hesitates.
    "commitment": 0.72,
}

MOMENTUM_COEF = {
    "exclamation": 0.48,
    "elongation": 0.44,
    "caps": 0.36,
    "emoji_energy": 0.34,
    "emoji_positive": 0.24,
    "laughter": 0.32,
    "lexicon_excitement": 0.50,
    "repetition": 0.22,
    "rate_spike": 0.50,
    "length_spike": 0.42,
    "interrobang": 0.30,
}

_POSITIVE_STEMS = frozenset({"yes", "yay", "yeah", "wow", "so", "hey", "woo", "omg"})
_NEGATIVE_STEMS = frozenset({"no", "nah", "ugh", "arg", "grr", "stop"})
_PARA = frozenset(
    {
        "laughter",
        "emoji_positive",
        "emoji_negative",
        "emoji_sarcasm",
        "emoji_energy",
        "exclamation",
        "question",
        "interrobang",
        "ellipsis",
        "caps",
        "elongation",
        "repetition",
        "filler",
        "pause",
        "hedge_strong",
        "hedge_weak",
        "self_correction",
        "sarcasm",
        "deflection",
        "dismissive",
        "negation",
        "rate_spike",
        "rate_drop",
        "length_spike",
        "overlap",
        "lexicon_frustration",
        "lexicon_excitement",
        "lexicon_confusion",
        "lexicon_relief",
        "lexicon_positive",
        "commitment",
    }
)


@dataclass
class EmotionBreakdown:
    state: EmotionalState
    per_turn: dict[int, dict[str, float]]


@dataclass
class DyadStats:
    speakers: list[str]
    jaccard: float
    echo: float
    shared_laughter: bool
    solo_laughter: bool
    rhythm: float
    backchannel_count: int
    repair_count: int
    friendly_overlap: bool
    responsive_answers: int


def _stem(word: str) -> str:
    """Collapse elongated letters so ``yesss`` can be compared with ``yes``."""
    out: list[str] = []
    index = 0
    lowered = word.lower()
    while index < len(lowered):
        char = lowered[index]
        run = index + 1
        while run < len(lowered) and lowered[run] == char:
            run += 1
        if run - index >= 3:
            out.append(char)
        else:
            out.append(lowered[index:run])
        index = run
    return "".join(out)


def _evidence(hit: Hit, weight: float) -> Evidence:
    span = Span(
        text=hit.excerpt[:80],
        start_char=hit.start_char,
        end_char=hit.end_char,
        speaker=hit.speaker,
        start=hit.start_s,
        end=hit.end_s,
        source="context" if hit.role == "context" else "current",
    )
    return Evidence(feature=hit.feature, detail=hit.detail, weight=unit(min(weight, 1.0)), span=span)


def _pattern(feature: str, detail: str, weight: float, turn: Turn | None = None) -> Evidence:
    span = None
    if turn is not None:
        span = Span(
            text=turn.text.strip()[:80],
            start_char=turn.base_char if turn.role == "current" and turn.base_char >= 0 else None,
            end_char=(turn.base_char + len(turn.text)) if turn.role == "current" and turn.base_char >= 0 else None,
            speaker=turn.speaker,
            start=turn.start,
            end=turn.end,
            source="context" if turn.role == "context" else "current",
        )
    return Evidence(feature=feature, detail=detail, weight=unit(min(weight, 1.0)), span=span)


def _top(evidence: list[Evidence], limit: int = 12) -> list[Evidence]:
    ranked = sorted(evidence, key=lambda item: (-item.weight, item.feature, item.detail))
    return ranked[:limit]


def evidence_confidence(
    tokens: int,
    evidence: list[Evidence],
    *,
    ceiling: float = 0.8,
    unknown_language: bool = False,
) -> float:
    """Confidence tracks how much observable material we had, and stops well below 1."""
    families = len({item.feature for item in evidence})
    mass = sum(item.weight for item in evidence)
    if tokens <= 2 and mass < 0.35:
        raw = 0.12 + 0.02 * max(tokens, 0)
    else:
        length = min(max(tokens, 0), 24) / 24
        raw = 0.14 + 0.32 * length + 0.07 * min(families, 5) + 0.05 * min(mass, 3.0)
        if tokens <= 3 and families <= 1:
            raw = min(raw, 0.30)
    if unknown_language:
        raw *= 0.85
    return unit(min(ceiling, raw))


def _emotion_contrib(hit: Hit, names: set[str]) -> dict[str, float]:
    feature = hit.feature
    weight = hit.weight
    if feature == "caps":
        hot = "lexicon_frustration" in names and not (names & {"laughter", "lexicon_excitement", "emoji_positive"})
        if hot:
            return {"frustration": 0.62 * weight}
        if names & {"laughter", "lexicon_excitement", "emoji_positive", "emoji_energy"}:
            return {"excitement": 0.58 * weight}
        return {"frustration": 0.20 * weight, "excitement": 0.26 * weight}
    if feature == "exclamation":
        hot = "lexicon_frustration" in names and not (names & {"laughter", "lexicon_excitement", "emoji_positive"})
        if hot:
            return {"frustration": 0.50 * weight, "excitement": 0.12 * weight}
        if names & {"laughter", "lexicon_excitement", "emoji_positive"}:
            return {"excitement": 0.58 * weight}
        return {"excitement": 0.48 * weight, "frustration": 0.10 * weight}
    if feature == "elongation":
        stem = _stem(hit.excerpt)
        if stem in _NEGATIVE_STEMS:
            return {"frustration": 0.55 * weight}
        if stem in _POSITIVE_STEMS or names & {"laughter", "lexicon_excitement"}:
            return {"excitement": 0.50 * weight}
        return {"excitement": 0.32 * weight, "frustration": 0.12 * weight}
    if feature == "repetition":
        token = hit.excerpt.split()[-1].lower() if hit.excerpt else ""
        if token in {"no", "nah", "never", "stop", "not"} or "negation" in names:
            return {"frustration": 0.16 * weight}
        if token in {"yes", "yeah", "yay", "okay", "ok"}:
            return {"excitement": 0.28 * weight}
        return {"frustration": 0.16 * weight, "excitement": 0.18 * weight}
    table: dict[str, dict[str, float]] = {
        "laughter": {"excitement": 0.42, "relief": 0.12},
        "emoji_positive": {"excitement": 0.40, "relief": 0.12},
        "emoji_negative": {"frustration": 0.55},
        "emoji_sarcasm": {"frustration": 0.28},
        "emoji_energy": {"excitement": 0.34},
        "question": {"confusion": 0.58},
        "interrobang": {"confusion": 0.42, "excitement": 0.32},
        "ellipsis": {"confusion": 0.30, "frustration": 0.10},
        "filler": {"confusion": 0.50},
        "pause": {"confusion": 0.40, "frustration": 0.10},
        "hedge_strong": {"confusion": 0.36},
        "hedge_weak": {"confusion": 0.16},
        "self_correction": {"confusion": 0.56},
        "lexicon_frustration": {"frustration": 0.82},
        "lexicon_excitement": {"excitement": 0.78},
        "lexicon_confusion": {"confusion": 0.80},
        "lexicon_relief": {"relief": 0.84},
        "repair": {"relief": 0.38},
        "dismissive": {"frustration": 0.22},
        "negation": {"frustration": 0.58},
        "rate_spike": {"excitement": 0.30, "frustration": 0.10},
        "rate_drop": {"confusion": 0.24},
        "sarcasm": {"frustration": 0.36},
        "deflection": {"frustration": 0.30},
    }
    coefs = table.get(feature)
    if not coefs:
        return {}
    return {emotion: coef * weight for emotion, coef in coefs.items()}


def _score_turn(hits: list[Hit], prior_frustration: float) -> tuple[dict[str, float], list[Evidence]]:
    names = {hit.feature for hit in hits}
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pieces: dict[str, list[tuple[Hit, str, float]]] = defaultdict(list)
    for hit in hits:
        for emotion, amount in _emotion_contrib(hit, names).items():
            if amount <= 0:
                continue
            grouped[hit.feature][emotion] += amount
            pieces[hit.feature].append((hit, emotion, amount))
    raw = {emotion: 0.0 for emotion in EMOTIONS}
    evidence: list[Evidence] = []
    for feature, emo_amounts in grouped.items():
        cap = FEATURE_CAP.get(feature, 1.5)
        total = sum(emo_amounts.values())
        scale = 1.0 if total <= cap else cap / total
        for hit, emotion, amount in pieces[feature]:
            scaled = amount * scale
            raw[emotion] += scaled
            if scaled >= 0.08:
                evidence.append(_evidence(hit, scaled))
    relief_trigger = "lexicon_relief" in names or "repair" in names or (
        "laughter" in names and "lexicon_frustration" not in names
    )
    if prior_frustration >= 0.42 and relief_trigger:
        raw["relief"] += 0.48
        evidence.append(
            Evidence(
                feature="relief_after_load",
                detail=f"softening after prior frustration {prior_frustration:.2f}",
                weight=unit(0.48),
            )
        )
    scores = {emotion: squash(raw[emotion]) for emotion in EMOTIONS}
    return scores, evidence


def _blend(per_turn: dict[int, dict[str, float]], turns: list[Turn]) -> dict[str, float]:
    if not turns:
        return {emotion: 0.0 for emotion in EMOTIONS}
    blended: dict[str, float] = {}
    for emotion in EMOTIONS:
        pairs = [(per_turn[turn.index][emotion], max(turn.token_count, 1)) for turn in turns]
        total = sum(weight for _, weight in pairs)
        average = sum(score * weight for score, weight in pairs) / total
        peak = max(score for score, _ in pairs)
        blended[emotion] = 0.55 * average + 0.45 * peak
    return blended


def _dominant(scores: dict[str, float]) -> str:
    ranked = sorted(EMOTIONS, key=lambda name: (-round(scores[name], 4), name))
    top = ranked[0]
    if scores[top] < DOMINANT_FLOOR:
        return "neutral"
    return top


def score_emotions(analysis: Analysis) -> EmotionBreakdown:
    per_turn: dict[int, dict[str, float]] = {}
    evidence: list[Evidence] = []
    prior_peak = 0.0
    for turn in analysis.doc.turns:
        scores, turn_evidence = _score_turn(analysis.by_turn.get(turn.index, []), prior_peak)
        per_turn[turn.index] = scores
        evidence.extend(turn_evidence)
        prior_peak = max(prior_peak, scores["frustration"])

    current = analysis.doc.current_turns()
    context = analysis.doc.context_turns()
    current_scores = _blend(per_turn, current)
    context_scores = _blend(per_turn, context)
    current_tokens = sum(turn.token_count for turn in current)
    context_tokens = sum(turn.token_count for turn in context)
    inherit = bool(context) and current_tokens < 6 and max(context_scores.values()) > max(current_scores.values())
    if not context:
        mix_current, mix_context = 1.0, 0.0
    elif inherit:
        mix_current, mix_context = 0.45, 0.55
    else:
        mix_current, mix_context = 0.82, 0.18
    blended = {
        emotion: mix_current * current_scores[emotion] + mix_context * context_scores[emotion] for emotion in EMOTIONS
    }
    if inherit:
        delta = max(blended.values()) - max(current_scores.values())
        if delta >= 0.08:
            evidence.append(
                Evidence(
                    feature="context_carry",
                    detail="short reply inherits prior emotional load",
                    weight=unit(min(delta, 1.0)),
                )
            )
    kept = _top(evidence)
    dominant = _dominant(blended)
    tokens_for_conf = current_tokens if current_tokens >= 6 else current_tokens + min(context_tokens, 16)
    confidence = evidence_confidence(
        tokens_for_conf,
        kept,
        unknown_language=analysis.doc.pack is None,
    )
    if dominant != "neutral":
        confidence = unit(max(confidence, min(0.8, 0.26 + 0.45 * blended[dominant])))
    elif current_tokens >= 12 and not context:
        confidence = unit(min(confidence, 0.48))
    state = EmotionalState(
        frustration=unit(blended["frustration"]),
        excitement=unit(blended["excitement"]),
        confusion=unit(blended["confusion"]),
        relief=unit(blended["relief"]),
        dominant=dominant,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=kept,
    )
    return EmotionBreakdown(state=state, per_turn=per_turn)


def _stopwords(analysis: Analysis) -> frozenset[str]:
    if analysis.doc.pack is None:
        return BASIC_STOPWORDS
    return analysis.doc.pack.stopwords | BASIC_STOPWORDS


def _turn_terms(turn: Turn, analysis: Analysis) -> set[str]:
    stops = _stopwords(analysis)
    if cjk_ratio(turn.text) >= 0.25:
        return char_bigrams(turn.text)
    return set(content_words(turn.text, stops))


def dyad_stats(analysis: Analysis) -> DyadStats:
    turns = analysis.doc.turns
    speakers = sorted({turn.speaker for turn in turns if turn.speaker})
    bags: dict[str, set[str]] = {speaker: set() for speaker in speakers}
    for turn in turns:
        if turn.speaker:
            bags[turn.speaker] |= _turn_terms(turn, analysis)
    pairwise: list[float] = []
    for index, left in enumerate(speakers):
        for right in speakers[index + 1 :]:
            pairwise.append(jaccard(bags[left], bags[right]))
    overlap = sum(pairwise) / len(pairwise) if pairwise else 0.0

    echoes: list[float] = []
    for prev, turn in zip(turns, turns[1:]):
        if not prev.speaker or not turn.speaker or prev.speaker == turn.speaker:
            continue
        current_terms = _turn_terms(turn, analysis)
        previous_terms = _turn_terms(prev, analysis)
        if not current_terms:
            continue
        echoes.append(len(current_terms & previous_terms) / len(current_terms))
    echo = sum(echoes) / len(echoes) if echoes else 0.0

    laugh_speakers: set[str] = set()
    any_laugh = False
    backchannels = 0
    repairs = 0
    friendly = False
    for turn in turns:
        names = {hit.feature for hit in analysis.by_turn.get(turn.index, [])}
        if "laughter" in names:
            any_laugh = True
            if turn.speaker:
                laugh_speakers.add(turn.speaker)
        backchannels += sum(1 for hit in analysis.by_turn.get(turn.index, []) if hit.feature == "backchannel")
        repairs += sum(1 for hit in analysis.by_turn.get(turn.index, []) if hit.feature == "repair")
        if "overlap" in names and "lexicon_frustration" not in names:
            friendly = True

    gaps: list[float] = []
    for prev, turn in zip(turns, turns[1:]):
        if prev.end is None or turn.start is None or turn.start < prev.end:
            continue
        gaps.append(turn.start - prev.end)
    rhythm = 0.0
    if len(gaps) >= 3:
        mean = statistics.fmean(gaps)
        if mean > 0.05:
            deviation = statistics.pstdev(gaps)
            rhythm = clamp(1.15 - (deviation / mean), 0.0, 1.0) * 0.85
    elif len(turns) >= 4:
        ratios: list[float] = []
        for prev, turn in zip(turns, turns[1:]):
            left = max(prev.token_count, 1)
            right = max(turn.token_count, 1)
            ratios.append(max(left, right) / min(left, right))
        if len(ratios) >= 3:
            smooth = sum(1 for ratio in ratios if ratio <= 2.2) / len(ratios)
            rhythm = 0.40 * smooth

    responsive = 0
    for prev, turn in zip(turns, turns[1:]):
        prev_question = any(hit.feature in {"question", "interrobang"} for hit in analysis.by_turn.get(prev.index, []))
        different = prev.speaker != turn.speaker or (prev.speaker is None and turn.speaker is None and prev.index != turn.index)
        if prev_question and different and turn.token_count >= 6 and (prev.speaker != turn.speaker):
            responsive += 1

    return DyadStats(
        speakers=speakers,
        jaccard=overlap,
        echo=echo,
        shared_laughter=len(laugh_speakers) >= 2,
        solo_laughter=any_laugh and len(laugh_speakers) < 2,
        rhythm=rhythm,
        backchannel_count=backchannels,
        repair_count=repairs,
        friendly_overlap=friendly,
        responsive_answers=responsive,
    )


def _quoted_positive(text: str, analysis: Analysis) -> bool:
    pack = analysis.doc.pack
    if pack is None:
        words = {"great", "fine", "sure", "perfect", "awesome", "love", "good"}
    else:
        words = {item.lower() for item in (*pack.positive, *pack.agreement)}
    for match in QUOTE_RE.finditer(text):
        inner = {token.lower() for token in match.group(1).split()}
        if inner & words:
            return True
    return False


def score_intent(analysis: Analysis, emotions: EmotionBreakdown, dyad: DyadStats) -> IntentAlignment:
    evidence: list[Evidence] = []
    turn_drifts: list[tuple[float, Turn, list[Evidence]]] = []
    turns = analysis.doc.turns
    for index, turn in enumerate(turns):
        if turn.role != "current":
            continue
        hits = analysis.by_turn.get(turn.index, [])
        names = {hit.feature for hit in hits}
        prior = [
            emotions.per_turn[earlier.index]["frustration"]
            for earlier in turns[:index]
            if earlier.index in emotions.per_turn
        ]
        prior_fr = max(prior) if prior else 0.0
        raw = 0.0
        local: list[Evidence] = []
        text = turn.text.strip()
        ends_flat = text.endswith((".", "。")) and "!" not in text and "！" not in text

        if "sarcasm" in names or "emoji_sarcasm" in names:
            raw += 0.75
            local.append(_pattern("sarcasm", "sarcasm marker conflicts with a straight reading", 0.75, turn))
        if _quoted_positive(turn.text, analysis):
            raw += 0.6
            local.append(_pattern("quoted_positive", "positive wording set in quotes", 0.6, turn))
        if "dismissive" in names and "repair" not in names:
            if turn.token_count <= 2 and prior_fr >= 0.38:
                raw += 1.2
                local.append(
                    _pattern(
                        "dismissive",
                        f'clipped reply after prior frustration {prior_fr:.2f}',
                        0.9,
                        turn,
                    )
                )
            elif turn.token_count <= 4 and prior_fr >= 0.38 and ends_flat:
                raw += 0.85
                local.append(_pattern("dismissive", "flat dismissive after heated prior turns", 0.75, turn))
            elif turn.token_count <= 2 and ends_flat:
                raw += 0.30
                local.append(_pattern("dismissive", "clipped flat reply with no elaboration", 0.30, turn))
        if "hedge_strong" in names and ("lexicon_positive" in names or "agreement" in names):
            raw += 0.55
            local.append(_pattern("hedged_positive", "positive or agreeing words wrapped in a hedge", 0.55, turn))

        prev = turns[index - 1] if index else None
        if (
            "agreement" in names
            and "laughter" not in names
            and "repair" not in names
            and turn.token_count <= 3
            and prev is not None
            and prev.speaker != turn.speaker
            and prev.token_count >= 8
            and prior_fr >= 0.40
        ):
            overlap = jaccard(_turn_terms(turn, analysis), _turn_terms(prev, analysis))
            if overlap < 0.08:
                raw += 0.45
                local.append(
                    _pattern(
                        "low_alignment_agreement",
                        "short agreement with almost no lexical overlap after a loaded turn",
                        0.45,
                        turn,
                    )
                )
        drift = squash(raw)
        turn_drifts.append((drift, turn, local))

    drift = max((item[0] for item in turn_drifts), default=0.0)
    drift_evidence = [item for score, _, local in turn_drifts if score >= DRIFT_DRIFTING for item in local]
    if not drift_evidence:
        drift_evidence = [item for _, _, local in turn_drifts for item in local]

    align_mass = 0.0
    align_evidence: list[Evidence] = []
    if dyad.jaccard >= 0.12:
        align_mass += clamp(dyad.jaccard / 0.22)
        align_evidence.append(
            Evidence(
                feature="mirroring",
                detail=f"cross-speaker lexical overlap {dyad.jaccard:.2f}",
                weight=unit(min(dyad.jaccard / 0.22, 1)),
            )
        )
    if dyad.shared_laughter and drift < DRIFT_DRIFTING:
        align_mass += 0.35
        align_evidence.append(Evidence(feature="laughter", detail="laughter shows up from more than one speaker", weight=0.35))
    if dyad.responsive_answers and drift < DRIFT_DRIFTING:
        align_mass += min(0.4, 0.2 * dyad.responsive_answers)
        align_evidence.append(
            Evidence(
                feature="responsive",
                detail=f"{dyad.responsive_answers} question(s) followed by a substantive reply",
                weight=unit(min(0.4, 0.2 * dyad.responsive_answers)),
            )
        )
    if dyad.repair_count and drift < DRIFT_DRIFTING:
        align_mass += 0.3
        align_evidence.append(Evidence(feature="repair", detail="repair or apology language", weight=0.3))

    state = emotions.state
    if drift < DRIFT_DRIFTING and state.frustration >= 0.4 and any(
        hit.feature == "lexicon_frustration" for hit in analysis.hits if hit.role == "current"
    ):
        align_mass += 0.35
        align_evidence.append(
            Evidence(
                feature="polarity_match",
                detail="frustration lexicon and frustration cues point the same way",
                weight=0.35,
            )
        )
    if drift < DRIFT_DRIFTING and state.excitement >= 0.4 and any(
        hit.feature in {"laughter", "lexicon_excitement", "exclamation", "elongation"}
        for hit in analysis.hits
        if hit.role == "current"
    ):
        align_mass += 0.30
        align_evidence.append(
            Evidence(
                feature="polarity_match",
                detail="excitement cues and energetic wording point the same way",
                weight=0.30,
            )
        )

    tokens = analysis.current_token_count
    if drift >= DRIFT_CONTRADICTORY:
        label = "contradictory"
        score = 1.0 - drift
        chosen = drift_evidence
    elif drift >= DRIFT_DRIFTING:
        label = "drifting"
        score = 1.0 - drift
        chosen = drift_evidence
    elif align_mass >= 0.25:
        label = "aligned"
        score = 0.60 + 0.35 * min(align_mass, 1.0)
        chosen = align_evidence
    elif tokens >= 20:
        label = "aligned"
        score = 0.64
        chosen = [
            Evidence(
                feature="no_drift_cues",
                detail="enough text to read, and no contradiction cues fired",
                weight=0.3,
            )
        ]
    else:
        label = "uncertain"
        score = 0.5
        chosen = []

    confidence = evidence_confidence(tokens + min(analysis.context_token_count, 12), chosen or drift_evidence, unknown_language=analysis.doc.pack is None)
    if label == "uncertain":
        confidence = unit(min(confidence, 0.22 + 0.01 * min(tokens, 8)))
    elif label == "contradictory":
        confidence = unit(max(confidence, min(0.86, 0.34 + 0.4 * drift)))
    return IntentAlignment(
        score=unit(score),
        drift=unit(drift),
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=_top(chosen),
    )


def _zones(analysis: Analysis, coefs: dict[str, float], floor: float) -> list[Zone]:
    zones: list[Zone] = []
    for turn in analysis.doc.current_turns():
        clauses = split_clauses(turn.text)
        if not clauses:
            continue
        hits = analysis.by_turn.get(turn.index, [])
        # Short turns read as one moment. Splitting "I was... in meetings" on the
        # ellipsis would name the zone after the fragment before the pause.
        if turn.token_count <= 16 or len(clauses) <= 1:
            clauses = [(0, len(turn.text), turn.text.strip())]
        # Attach rate/length once, to the hottest clause, so a whole-turn cue does not stamp every sentence.
        mobile = {"rate_spike", "length_spike"}
        mobile_hits = [hit for hit in hits if hit.feature in mobile and hit.feature in coefs]
        fixed_hits = [hit for hit in hits if hit.feature not in mobile]
        drafted: list[tuple[float, int, int, str, list[Hit]]] = []
        for start, end, text in clauses:
            local_hits = [hit for hit in fixed_hits if hit.feature in coefs and start <= hit.local_start < end]
            if start == 0:
                local_hits.extend(
                    hit
                    for hit in fixed_hits
                    if hit.feature in {"pause", "rate_drop", "overlap"}
                    and hit.feature in coefs
                    and hit.local_start == 0
                    and hit not in local_hits
                )
            raw = sum(coefs[hit.feature] * hit.weight for hit in local_hits)
            drafted.append((raw, start, end, text, local_hits))
        if mobile_hits and drafted:
            hottest = max(range(len(drafted)), key=lambda index: drafted[index][0])
            raw, start, end, text, local_hits = drafted[hottest]
            local_hits = [*local_hits, *mobile_hits]
            drafted[hottest] = (raw + sum(coefs[hit.feature] * hit.weight for hit in mobile_hits), start, end, text, local_hits)
        for raw, start, end, text, local_hits in drafted:
            if not local_hits:
                continue
            intensity = squash(raw)
            if intensity < floor:
                continue
            stops = _stopwords(analysis)
            evidence = [_evidence(hit, coefs[hit.feature] * hit.weight) for hit in local_hits]
            families = len({hit.feature for hit in local_hits})
            zones.append(
                Zone(
                    topic=topic_phrase(text, stops),
                    span=Span(
                        text=text[:180],
                        start_char=turn.base_char + start,
                        end_char=turn.base_char + end,
                        speaker=turn.speaker,
                        start=turn.start,
                        end=turn.end,
                        source="current",
                    ),
                    intensity=unit(intensity),
                    confidence=unit(min(0.8, 0.24 + 0.12 * families + 0.25 * intensity)),
                    evidence=_top(evidence, limit=8),
                )
            )
    zones.sort(key=lambda zone: (-zone.intensity, zone.span.start_char or 0, zone.topic))
    return zones


def pressure_zones(analysis: Analysis) -> list[Zone]:
    """Spans where hesitation clusters, or a commitment/money ask lands."""
    return _zones(analysis, PRESSURE_COEF, PRESSURE_FLOOR)


def momentum_zones(analysis: Analysis) -> list[Zone]:
    """Spans where rate, length, exclamation, or other energy rises together."""
    return _zones(analysis, MOMENTUM_COEF, MOMENTUM_FLOOR)


def score_connection(analysis: Analysis, dyad: DyadStats) -> ConnectionScore:
    laughter = 0.9 if dyad.shared_laughter else (0.28 if dyad.solo_laughter else 0.0)
    mirroring = clamp(dyad.jaccard / 0.18) * 0.75 + clamp(dyad.echo / 0.40) * 0.25
    backchannel = clamp(dyad.backchannel_count / 2)
    raw = 0.0
    if dyad.shared_laughter:
        raw += 1.15
    elif dyad.solo_laughter:
        raw += 0.32
    raw += 0.90 * mirroring
    raw += 0.50 * dyad.rhythm
    raw += 0.28 * backchannel
    if dyad.friendly_overlap:
        raw += 0.18
    evidence: list[Evidence] = []
    if laughter:
        evidence.append(
            Evidence(
                feature="laughter",
                detail="shared laughter" if dyad.shared_laughter else "laughter from one side",
                weight=unit(laughter),
            )
        )
    if mirroring >= 0.15:
        evidence.append(
            Evidence(
                feature="mirroring",
                detail=f"lexical overlap {dyad.jaccard:.2f}, echo {dyad.echo:.2f}",
                weight=unit(min(mirroring, 1)),
            )
        )
    if dyad.rhythm >= 0.2:
        evidence.append(Evidence(feature="rhythm", detail="turn timing or length is relatively steady", weight=unit(dyad.rhythm)))
    if backchannel >= 0.2:
        evidence.append(
            Evidence(
                feature="backchannel",
                detail=f"{dyad.backchannel_count} short acknowledgement(s)",
                weight=unit(backchannel),
            )
        )
    for hit in analysis.hits:
        if hit.feature == "laughter" and len(evidence) < 8:
            evidence.append(_evidence(hit, 0.4 if dyad.shared_laughter else 0.25))
    tokens = analysis.current_token_count + analysis.context_token_count
    confidence = evidence_confidence(tokens, evidence)
    if len(dyad.speakers) < 2:
        confidence = unit(confidence * 0.5)
    if tokens < 8:
        confidence = unit(confidence * 0.75)
    return ConnectionScore(
        score=unit(squash(raw)),
        confidence=confidence,
        components=ConnectionComponents(
            laughter=unit(laughter),
            mirroring=unit(mirroring),
            rhythm=unit(dyad.rhythm),
            backchannel=unit(backchannel),
        ),
        evidence=_top(evidence),
    )


def score_signal_strength(analysis: Analysis, drift: float = 0.0) -> SignalStrength:
    hits = [hit for hit in analysis.hits if hit.role == "current"]
    mass = 0.0
    for hit in hits:
        if hit.feature == "dismissive":
            # A lone "ok" is a weak pragmatic mark, not a dense signal.
            mass += 0.25 * hit.weight
        elif hit.feature in _PARA:
            mass += hit.weight
        elif hit.feature in {"agreement", "repair", "backchannel"}:
            mass += 0.15 * hit.weight
    # A clipped "fine." is short, but the mismatch with prior turns is the signal.
    if drift >= 0.34:
        mass += 0.9 + 0.6 * drift
    tokens = analysis.current_token_count
    score = squash(0.22 * min(mass, 12.0))
    if tokens <= 2 and mass < 0.5 and drift < 0.34:
        score *= 0.30
    families = len({hit.feature for hit in hits if hit.feature in _PARA})
    evidence = [_evidence(hit, min(hit.weight, 1.0)) for hit in hits if hit.feature in _PARA]
    if drift >= 0.34:
        evidence.append(
            Evidence(
                feature="intent_drift",
                detail=f"wording diverges from the surrounding cues ({drift:.2f})",
                weight=unit(min(drift, 1.0)),
            )
        )
    confidence = evidence_confidence(tokens, _top(evidence))
    if tokens <= 2 and mass < 0.8:
        confidence = unit(min(confidence, 0.16))
    elif mass < 0.3:
        confidence = unit(min(confidence, 0.18 + 0.015 * min(tokens, 16)))
    else:
        confidence = unit(max(confidence, min(0.8, 0.2 + 0.07 * families)))
    return SignalStrength(score=unit(score), confidence=confidence, evidence=_top(evidence))


def choose_style(
    analysis: Analysis,
    emotions: EmotionBreakdown,
    intent: IntentAlignment,
    connection: ConnectionScore,
    pressure: list[Zone],
    momentum: list[Zone],
    signal: SignalStrength,
) -> RecommendedResponseStyle:
    """Pick one stance. The order is the policy: confusion, then heat or drift, then play, then hype."""
    state = emotions.state
    frustration = state.frustration
    excitement = state.excitement
    confusion = state.confusion
    relief = state.relief
    drift = intent.drift
    pressure_max = max((zone.intensity for zone in pressure), default=0.0)
    momentum_max = max((zone.intensity for zone in momentum), default=0.0)
    tokens = analysis.current_token_count

    def finish(style: ResponseStyle, trigger: float, detail: str) -> RecommendedResponseStyle:
        if style == "direct" and signal.score < 0.25:
            confidence = min(0.42, 0.12 + 0.45 * signal.confidence)
            if tokens < 4:
                confidence = min(confidence, 0.24)
        else:
            confidence = min(0.8, 0.32 + 0.5 * trigger)
        if analysis.doc.pack is None:
            confidence *= 0.9
        return RecommendedResponseStyle(
            style=style,
            confidence=unit(confidence),
            evidence=[Evidence(feature="style_rule", detail=detail, weight=unit(min(max(trigger, 0.05), 1)))],
        )

    if confusion >= 0.42 and confusion >= frustration - 0.04 and confusion >= excitement - 0.08:
        return finish(
            "clarifying",
            confusion,
            f"confusion {confusion:.2f} leads frustration {frustration:.2f} and excitement {excitement:.2f}",
        )
    if confusion >= 0.36 and pressure_max >= 0.42 and confusion >= excitement and confusion >= frustration - 0.08:
        return finish(
            "clarifying",
            confusion,
            f"confusion {confusion:.2f} clusters with pressure {pressure_max:.2f}",
        )
    if frustration >= 0.40 and frustration >= excitement - 0.02:
        return finish("calm", frustration, f"frustration {frustration:.2f} is the leading load")
    if drift >= 0.50:
        return finish("calm", drift, f"intent drift {drift:.2f}, so the surface wording is a poor guide")
    if (
        frustration < 0.30
        and excitement < 0.58
        and connection.score >= 0.42
        and connection.components.laughter >= 0.55
    ):
        return finish(
            "playful",
            connection.score,
            f"shared warmth {connection.score:.2f} without a frustration spike",
        )
    if frustration < 0.28 and relief >= 0.48 and excitement < 0.50:
        style: ResponseStyle = "playful" if connection.score >= 0.30 else "calm"
        return finish(style, relief, f"relief {relief:.2f} after the heavier moment")
    if excitement >= 0.48 and excitement >= frustration and excitement >= confusion - 0.05:
        return finish("hype", excitement, f"excitement {excitement:.2f} is the leading load")
    if momentum_max >= 0.55 and excitement >= 0.34 and frustration < 0.32 and excitement >= confusion:
        return finish("hype", momentum_max, f"momentum {momentum_max:.2f} with rising excitement")
    return finish(
        "direct",
        max(signal.score, 0.2),
        "no strong emotional lead — stay plain and specific",
    )
