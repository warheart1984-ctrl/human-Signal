"""Pluggable language packs.

Core detectors do not import these tables except through :func:`resolve_language`.
An unknown code yields ``None`` and compilation continues on punctuation, timing,
laughter tokens, repetition, and overlap.

To add a language, build a :class:`LanguagePack` and call :func:`register_pack`.
Packs with ``use_word_boundaries`` False (Japanese) match by substring because
the script does not split words with spaces. Prefer longer phrases there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_EN_FUNCTION = frozenset(
    {
        "the",
        "and",
        "you",
        "that",
        "this",
        "with",
        "have",
        "was",
        "were",
        "are",
        "for",
        "just",
        "it's",
        "its",
        "but",
        "not",
        "your",
        "from",
        "they",
        "them",
        "what",
        "when",
        "this",
    }
)
_ES_FUNCTION = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "que",
        "del",
        "una",
        "por",
        "con",
        "para",
        "está",
        "esta",
        "como",
        "pero",
        "porque",
        "hay",
        "sus",
        "eso",
        "esa",
        "no",
        "es",
        "lo",
        "un",
        "al",
    }
)


@dataclass(frozen=True)
class LanguagePack:
    """Lexicon for one language. Phrases are lowercase."""

    code: str
    name: str
    use_word_boundaries: bool
    frustration_strong: tuple[str, ...] = ()
    frustration_weak: tuple[str, ...] = ()
    excitement_strong: tuple[str, ...] = ()
    excitement_weak: tuple[str, ...] = ()
    confusion: tuple[str, ...] = ()
    relief: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    agreement: tuple[str, ...] = ()
    hedge_strong: tuple[str, ...] = ()
    hedge_weak: tuple[str, ...] = ()
    fillers: tuple[str, ...] = ()
    self_corrections: tuple[str, ...] = ()
    leading_self_corrections: tuple[str, ...] = ()
    deflection: tuple[str, ...] = ()
    dismissives: tuple[str, ...] = ()
    repair: tuple[str, ...] = ()
    sarcasm: tuple[str, ...] = ()
    backchannels: tuple[str, ...] = ()
    negation: tuple[str, ...] = ()
    # Money and commitment asks. Strong phrases can open a pressure zone alone.
    # Weak ones (bare "pay", "sign") only add weight, so a slogan does not.
    commitment_strong: tuple[str, ...] = ()
    commitment_weak: tuple[str, ...] = ()
    # Tokens like "again" that are neutral unless negated, "yet"-marked, or punctuated.
    loaded_repeats: tuple[str, ...] = ()
    stopwords: frozenset[str] = frozenset()

    def categories(self) -> dict[str, tuple[str, ...]]:
        return {
            "frustration_strong": self.frustration_strong,
            "frustration_weak": self.frustration_weak,
            "excitement_strong": self.excitement_strong,
            "excitement_weak": self.excitement_weak,
            "confusion": self.confusion,
            "relief": self.relief,
            "positive": self.positive,
            "agreement": self.agreement,
            "hedge_strong": self.hedge_strong,
            "hedge_weak": self.hedge_weak,
            "filler": self.fillers,
            "self_correction": self.self_corrections,
            "deflection": self.deflection,
            "repair": self.repair,
            "sarcasm": self.sarcasm,
            "backchannel": self.backchannels,
            "commitment_strong": self.commitment_strong,
            "commitment_weak": self.commitment_weak,
        }


PACKS: dict[str, LanguagePack] = {}

_EN_STOP = frozenset(
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
        "been",
        "being",
        "because",
        "while",
        "after",
        "before",
        "there",
        "here",
        "also",
        "very",
        "let",
        "got",
        "get",
    }
)

EN = LanguagePack(
    code="en",
    name="English",
    use_word_boundaries=True,
    frustration_strong=(
        "ridiculous",
        "unfair",
        "furious",
        "frustrated",
        "frustrating",
        "annoyed",
        "sick of",
        "tired of",
        "fed up",
        "not listening",
        "you never",
        "you always",
        "you're always",
        "you are always",
        "nothing happened",
        "still haven't",
        "haven't looked",
        "third time",
        "come on",
        "so done",
        "i'm done",
        "i am done",
        "unacceptable",
        "you ignored",
        "ignored me",
        "this is ridiculous",
    ),
    frustration_weak=("seriously", "still"),
    excitement_strong=(
        "so excited",
        "let's go",
        "hell yes",
        "love this",
        "can't wait",
        "cannot wait",
        "hyped",
        "obsessed",
        "amazing",
        "incredible",
    ),
    excitement_weak=("love", "awesome", "wow"),
    confusion=(
        "confused",
        "confusing",
        "don't understand",
        "do not understand",
        "don't get it",
        "what do you mean",
        "wait what",
        "i'm lost",
        "i am lost",
        "lost me",
        "not sure i follow",
        "huh",
    ),
    relief=(
        "phew",
        "thank god",
        "thank goodness",
        "what a relief",
        "glad that's",
        "at last",
        "finally",
    ),
    positive=(
        "great",
        "awesome",
        "perfect",
        "wonderful",
        "happy",
        "fine",
        "good",
        "excellent",
        "love",
        "nice",
    ),
    agreement=(
        "yes",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "sure",
        "right",
        "totally",
        "agreed",
        "exactly",
        "of course",
        "sounds good",
    ),
    hedge_strong=(
        "i guess",
        "sort of",
        "kind of",
        "kinda",
        "not sure",
        "maybe",
        "perhaps",
        "i suppose",
    ),
    hedge_weak=("probably", "i think", "a bit", "somewhat", "possibly", "a little"),
    fillers=("er", "hmm"),
    self_corrections=(
        "i mean",
        "i meant",
        "let me rephrase",
        "or rather",
        "wait no",
        "sorry no",
        "hold on",
        "actually no",
    ),
    leading_self_corrections=("wait",),
    deflection=(
        "never mind",
        "nevermind",
        "moving on",
        "forget it",
        "it is what it is",
        "doesn't matter",
        "does not matter",
    ),
    dismissives=(
        "fine",
        "sure",
        "whatever",
        "ok",
        "okay",
        "great",
        "cool",
        "perfect",
        "right",
        "yeah",
        "yep",
        "totally",
        "k",
    ),
    repair=(
        "i'm sorry",
        "i am sorry",
        "my fault",
        "i promise",
        "you're right",
        "you are right",
        "i hear you",
        "fair enough",
    ),
    sarcasm=("yeah right", "as if", "sure jan"),
    backchannels=(
        "yeah",
        "yep",
        "right",
        "exactly",
        "mhm",
        "uh-huh",
        "mm-hmm",
        "totally",
        "nice",
        "cool",
        "ok",
        "okay",
    ),
    negation=("no", "not", "never", "stop", "nope", "nah"),
    commitment_strong=(
        "paid",
        "payment",
        "payments",
        "transaction",
        "transactions",
        "invoice",
        "invest",
        "investment",
        "investing",
        "deadline",
        "deposit",
        "escrow",
        "subscription",
    ),
    commitment_weak=(
        "pay",
        "paying",
        "rent",
        "budget",
        "price",
        "charge",
        "sign",
        "signing",
        "contract",
        "fee",
        "fees",
        "cost",
        "costs",
    ),
    loaded_repeats=("again",),
    stopwords=_EN_STOP,
)

_ES_STOP = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "que",
        "del",
        "una",
        "por",
        "con",
        "para",
        "está",
        "esta",
        "como",
        "pero",
        "porque",
        "hay",
        "sus",
        "eso",
        "esa",
        "ese",
        "unos",
        "unas",
        "les",
        "nos",
        "más",
        "mas",
        "muy",
        "ya",
        "su",
        "al",
        "se",
        "lo",
        "le",
        "me",
        "te",
        "tu",
        "es",
        "en",
        "un",
        "de",
        "y",
        "o",
        "mi",
    }
)

ES = LanguagePack(
    code="es",
    name="Spanish",
    use_word_boundaries=True,
    frustration_strong=(
        "no puede ser",
        "es injusto",
        "me tienes harto",
        "me tienes harta",
        "siempre igual",
        "otra vez",
        "ya basta",
        "no me escuchas",
        "ni me miras",
        "estás loco",
        "estas loco",
        "qué rabia",
        "que rabia",
    ),
    frustration_weak=("siempre", "nunca", "en serio"),
    excitement_strong=(
        "me encanta",
        "qué bien",
        "que bien",
        "buenísimo",
        "buenisimo",
        "vamos",
        "no puedo esperar",
    ),
    excitement_weak=("genial", "increíble", "increible"),
    confusion=(
        "no entiendo",
        "no me queda claro",
        "a qué te refieres",
        "cómo que",
        "como que",
        "no sé qué",
        "no se que",
    ),
    relief=(
        "menos mal",
        "qué alivio",
        "que alivio",
        "por fin",
        "gracias a dios",
        "uff",
        "uf",
    ),
    positive=("bien", "genial", "perfecto", "bueno", "excelente", "feliz"),
    agreement=("sí", "si", "vale", "claro", "ok", "okay", "dale", "bueno", "exacto", "de acuerdo"),
    hedge_strong=("no sé", "no se", "quizás", "quizas", "tal vez", "puede ser", "a lo mejor"),
    hedge_weak=("creo que", "algo así", "algo asi", "probablemente"),
    fillers=("este", "pues", "eh", "em"),
    self_corrections=("o sea", "quiero decir", "mejor dicho", "a ver", "o mejor"),
    leading_self_corrections=("espera",),
    deflection=("como sea", "da igual", "olvídalo", "olvidalo", "en fin", "ya ni modo"),
    dismissives=("bien", "vale", "claro", "bueno", "sí", "si", "ok", "dale", "genial", "perfecto"),
    repair=("lo siento", "perdón", "perdon", "tienes razón", "tienes razon", "prometo", "fue mi culpa"),
    sarcasm=("sí claro", "si claro", "ya seguro"),
    backchannels=("sí", "si", "vale", "claro", "exacto", "ajá", "aja", "ok"),
    negation=("no", "nunca", "ni"),
    commitment_strong=(
        "pago",
        "pagar",
        "transacción",
        "transaccion",
        "factura",
        "invertir",
        "inversión",
        "inversion",
        "depósito",
        "deposito",
        "presupuesto",
    ),
    commitment_weak=("firmar", "plazo", "renta", "cobrar"),
    stopwords=_ES_STOP,
)

_JA_STOP = frozenset(
    {
        "です",
        "ます",
        "する",
        "した",
        "して",
        "これ",
        "それ",
        "あれ",
        "この",
        "その",
        "あの",
    }
)

JA = LanguagePack(
    code="ja",
    name="Japanese",
    use_word_boundaries=False,
    frustration_strong=(
        "ありえない",
        "むかつく",
        "ふざけ",
        "ひどい",
        "またか",
        "聞いてない",
        "最悪",
        "信じられない",
        "もういい",
    ),
    frustration_weak=("いつも", "絶対"),
    excitement_strong=("最高", "やった", "楽しみ", "すごくいい", "めっちゃいい", "ワクワク"),
    excitement_weak=("いいね", "すごい", "好き"),
    confusion=(
        "どういうこと",
        "わからない",
        "分からない",
        "意味がわからない",
        "よくわからない",
        "どういう意味",
    ),
    relief=("よかった", "助かった", "ほっとした", "やっと"),
    positive=("大丈夫", "よかった", "最高", "好き"),
    agreement=("うん", "はい", "わかった", "了解", "オーケー", "オッケー", "確かに"),
    hedge_strong=("かもしれない", "かもね", "多分", "たぶん", "自信ない"),
    hedge_weak=("と思う", "ちょっと", "一応"),
    fillers=("えっと", "えーと", "えー", "うーん", "なんか", "そのー", "あの"),
    self_corrections=("というか", "ていうか", "じゃなくて", "ではなく", "やっぱり"),
    leading_self_corrections=(),
    deflection=("どうでもいい", "とにかく", "別にいい"),
    dismissives=("別に", "いいよ", "どうでも", "わかった", "はいはい", "大丈夫"),
    repair=("ごめん", "すみません", "すまない", "悪かった"),
    sarcasm=("はいはい",),
    backchannels=("うんうん", "なるほど", "そうだね", "たしかに", "確かに", "へえ"),
    negation=("じゃない", "違う", "ちがう"),
    commitment_strong=("支払い", "送金", "契約", "締め切り", "投資"),
    commitment_weak=("予算", "料金"),
    stopwords=_JA_STOP,
)


def register_pack(pack: LanguagePack) -> None:
    """Install or replace a pack. Safe to call at import or from tests."""
    PACKS[pack.code] = pack
    _phrase_patterns.cache_clear()


def unregister_pack(code: str) -> None:
    """Remove a pack. Built-ins can be re-registered by importing them again."""
    PACKS.pop(code, None)
    _phrase_patterns.cache_clear()


_ALIASES = {"eng": "en", "spa": "es", "es": "es", "jpn": "ja", "jp": "ja", "en": "en", "ja": "ja"}


@lru_cache(maxsize=32)
def _phrase_patterns(code: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """One pattern per phrase so overlapping candidates can all be seen.

    A single alternation would commit to the earliest match and hide a longer
    phrase that starts a few characters later (``still haven't`` vs ``haven't looked``).
    """
    pack = PACKS[code]
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for category, phrases in pack.categories().items():
        for phrase in phrases:
            cleaned = phrase.strip()
            if not cleaned:
                continue
            compiled.append((category, re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)")))
    return tuple(compiled)


def phrase_patterns(pack: LanguagePack) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if pack.code not in PACKS or PACKS[pack.code] is not pack:
        register_pack(pack)
    return _phrase_patterns(pack.code)


def _count_function_words(text: str, words: frozenset[str]) -> int:
    count = 0
    for match in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower()):
        if match.group(0) in words:
            count += 1
    return count


def detect_language(text: str) -> str:
    """Lightweight script and function-word sniff. Returns ``en``, ``es``, ``ja``, or ``unknown``."""
    kana = len(re.findall(r"[ぁ-んァ-ン]", text))
    if kana >= 3:
        return "ja"
    marks = text.count("¿") + text.count("¡")
    english = _count_function_words(text, _EN_FUNCTION)
    spanish = _count_function_words(text, _ES_FUNCTION) + marks * 2
    if spanish >= 2 and spanish > english:
        return "es"
    if english >= 2 and english >= spanish:
        return "en"
    return "unknown"


def resolve_language(explicit: str | None, text: str) -> tuple[str, LanguagePack | None]:
    """Return ``(code, pack)``. ``pack`` is None when the code has no lexicon."""
    if explicit:
        raw = explicit.strip().lower()
        primary = raw.split("-", 1)[0]
        code = _ALIASES.get(primary, primary)
        return code, PACKS.get(code)
    detected = detect_language(text)
    return detected, PACKS.get(detected)


register_pack(EN)
register_pack(ES)
register_pack(JA)
