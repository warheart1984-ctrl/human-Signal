"""Language-independent text measurements.

These helpers look at shape (marks, repetition, timing tokens), not meaning.
Lexicons live in :mod:`humansignal.languages`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words, numbers, bracket markers, kana runs, kanji, hangul.
TOKEN_RE = re.compile(
    r"\[[^\[\]\n]{1,48}\]"
    r"|[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?"
    r"|[0-9]+(?:\.[0-9]+)?"
    r"|[ぁ-んァ-ンー]{1,12}"
    r"|[一-龯々]"
    r"|[가-힣]{1,8}"
)

LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ]+)?")

# Laughter tokens that show up across languages. Deliberately structural:
# repeated syllables, not a sentiment dictionary.
LAUGHTER_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:ha){2,}h?\b"
    r"|\b(?:he){2,}h?\b"
    r"|\b(?:hi){2,}h?\b"
    r"|\b(?:ja){2,}j?\b"
    r"|\b(?:je){2,}j?\b"
    r"|\bahaha+\b"
    r"|\blo+l\b"
    r"|\blmao\b"
    r"|\blmfao\b"
    r"|\brofl\b"
    r"|\bmdr\b"
    r"|\bptdr\b"
    r"|\bk{3,}\b"
    r"|\brs(?:rs)+\b"
    r"|\bksks+\b"
    r"|\bhuehue+\b"
    r"|(?<![:/\w])w{3,}(?![\w.])"
    r"|(?<!\d)5{3,}(?!\d)"
    r"|哈哈|呵呵|嘿嘿|嘻嘻|笑{2,}|（笑）|\(笑\)"
    r"|ㅋㅋ+|ㅎㅎ+|하하+|헤헤+"
    r")"
)

CJK_OR_HANGUL_LAUGH_RE = re.compile(r"哈哈|呵呵|嘿嘿|嘻嘻|笑{2,}|（笑）|\(笑\)|ㅋㅋ+|ㅎㅎ+|하하+|헤헤+")

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+"
)

ELLIPSIS_RE = re.compile(r"\.{3,}|…+")
QUESTION_RE = re.compile(r"[?？]+")
EXCLAIM_RE = re.compile(r"[!！]+")
INTERROBANG_RE = re.compile(r"[!！][?？]|[?？][!！]")
CAPS_RE = re.compile(r"\b[A-ZÀ-ÖØ-Þ]{2,}(?:'[A-ZÀ-ÖØ-Þ]+)?\b")
# A run of 3+ identical letters is only a candidate. ``telll`` is one extra
# consonant on a doubled spelling; emphatic stretch is decided in
# :func:`is_emphatic_elongation`.
ELONGATION_RE = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ])\1{2,}")
_ELONGATION_VOWELS = frozenset("aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ")
# Stems people actually stretch (``yesss`` → ``yes``). A consonant triple whose
# stem is not here is treated as a typo, not emphasis.
_ELONGATABLE_STEMS = frozenset(
    {
        "yes",
        "yay",
        "yeah",
        "yo",
        "so",
        "no",
        "nah",
        "hey",
        "hi",
        "wow",
        "woo",
        "omg",
        "oh",
        "ah",
        "ow",
        "aw",
        "stop",
        "wait",
        "please",
        "love",
        "cool",
        "ok",
        "okay",
        "bye",
        "ugh",
        "arg",
        "grr",
        "whoa",
        "damn",
        "really",
        "super",
        "go",
        "now",
        "what",
        "why",
    }
)
MARKER_RE = re.compile(r"\[([^\[\]\n]{1,48})\]")
CORE_FILLER_RE = re.compile(r"(?i)(?<!\w)(?:uh+|um+|erm+)(?!\w)")
SARCASM_SLASH_RE = re.compile(r"(?i)(?:^|\s)/s\b")
QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{2,40})[\"“”]")
EMOTICON_POS_RE = re.compile(r"(?:(?<!\w)[:;=]-?[)D]|<3|\bxd\b)", re.IGNORECASE)
EMOTICON_NEG_RE = re.compile(r"(?<!\w)[:;=]-?[(]")

ACRONYMS = frozenset(
    {
        "OK",
        "API",
        "URL",
        "URI",
        "JSON",
        "CPU",
        "GPU",
        "HTTP",
        "HTTPS",
        "HTML",
        "CSS",
        "SQL",
        "SSH",
        "AWS",
        "GCP",
        "ID",
        "UID",
        "UUID",
        "LLM",
        "NLP",
        "TTS",
        "ASR",
        "EOF",
        "NULL",
        "TODO",
        "FIXME",
        "PR",
        "CI",
        "CD",
    }
)

# Used when no language pack is active, so function words do not look like topics.
BASIC_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "was",
        "were",
        "are",
        "for",
        "you",
        "your",
        "but",
        "not",
        "its",
        "it's",
        "just",
        "into",
        "about",
        "then",
        "than",
        "they",
        "them",
        "she",
        "his",
        "her",
        "our",
        "out",
        "que",
        "los",
        "las",
        "del",
        "una",
        "por",
        "con",
        "para",
    }
)

TOPIC_SKIP = BASIC_STOPWORDS | frozenset(
    {
        "wait",
        "like",
        "just",
        "really",
        "think",
        "guess",
        "mean",
        "meant",
        "hold",
        "actually",
        "maybe",
        "probably",
        "something",
        "anything",
        "everything",
        "here",
        "there",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "how",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "does",
        "did",
        "doing",
        "been",
        "being",
        "because",
        "while",
        "after",
        "before",
        "again",
        "still",
        "already",
        "very",
        "also",
        "let",
        "lets",
        "i'm",
        "i've",
        "i'll",
        "i'd",
        "don't",
        "can't",
        "that's",
        "it's",
        "you're",
        "we're",
        "they're",
        "didn't",
        "doesn't",
        "isn't",
        "wasn't",
        "won't",
    }
)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def is_marker(self) -> bool:
        return self.text.startswith("[") and self.text.endswith("]")


def normalize_match_text(text: str) -> str:
    """Lowercase and unify apostrophes without changing string length."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
    )


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def word_tokens(tokens: list[Token]) -> list[Token]:
    return [tok for tok in tokens if not tok.is_marker]


def letter_count(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


def elongation_stem(word: str) -> str:
    """Collapse runs of 3+ identical letters so ``yesss`` compares with ``yes``."""
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


def is_emphatic_elongation(word: str) -> bool:
    """True when a repeated-letter run is emphasis, not a one-letter typo.

    Counts a vowel stretched 3+ times (``sooo``), any letter stretched 4+ times
    (past a single extra keystroke), or a known elongatable stem (``yesss``).
    ``telll`` and ``helllo`` fail all three: the extra consonant is a typo.
    """
    lowered = word.lower()
    runs: list[tuple[str, int]] = []
    index = 0
    while index < len(lowered):
        char = lowered[index]
        run = index + 1
        while run < len(lowered) and lowered[run] == char:
            run += 1
        if run - index >= 3 and char.isalpha():
            runs.append((char, run - index))
        index = run
    if not runs:
        return False
    if any(length >= 4 for _, length in runs):
        return True
    if any(char in _ELONGATION_VOWELS for char, _ in runs):
        return True
    return elongation_stem(word) in _ELONGATABLE_STEMS


def overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= left or start >= right) for left, right in spans)


def split_clauses(text: str) -> list[tuple[int, int, str]]:
    """Split a turn into sentence-like clauses, preserving offsets."""
    stripped = text.strip()
    if not stripped:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    # A closing quote does not glue two sentences together: `output." Keep` still splits.
    for match in re.finditer(r"[.!?。！？…]+[\"“”']*(?:\s+|$)|(?:\n+)", text):
        end = match.end()
        chunk = text[start:end]
        if chunk.strip():
            trimmed = chunk.rstrip()
            while trimmed and trimmed[-1] in "\"“”'":
                trimmed = trimmed[:-1].rstrip()
            spans.append((start, start + len(trimmed), trimmed.strip()))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text), text[start:].strip()))
    if not spans:
        spans.append((0, len(text), stripped))
    return spans


def content_words(text: str, stopwords: frozenset[str]) -> list[str]:
    """Ordered unique content words for topics and lexical overlap."""
    seen: list[str] = []
    known: set[str] = set()
    for match in LATIN_WORD_RE.finditer(text):
        word = match.group(0).lower()
        if len(word) <= 2 or word in stopwords or word in known:
            continue
        known.add(word)
        seen.append(word)
    return seen


def char_bigrams(text: str) -> set[str]:
    chars = [ch for ch in text if not ch.isspace() and ch not in "。、！？!?.,\"'()[]"]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(re.findall(r"[ぁ-んァ-ン一-龯々가-힣]", text))
    return cjk / max(len(text), 1)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def topic_phrase(text: str, stopwords: frozenset[str]) -> str:
    text = MARKER_RE.sub(" ", text)
    words = [w for w in content_words(text, stopwords | TOPIC_SKIP) if w not in TOPIC_SKIP]
    if words:
        return " ".join(words[:5])
    cleaned = MARKER_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n.!?…。！？")
    if not cleaned:
        return "unspecified"
    return cleaned[:48]
