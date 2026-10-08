# HumanSignal

HumanSignal is a small **emotional compiler**. It turns a messy human turn — a chat line, or a timed transcript — into structured JSON: emotional load, whether the words match the other cues, where the speaker hesitates, where energy rises, and how connected the speakers look.

This is signal architecture, not therapy and not a sentiment label. Sentiment asks "is this positive?" HumanSignal asks which observable cues fired, how strong they are, and how much the compiler should trust them. It does not read minds, diagnose anyone, or tell you the truth of a relationship. Every score carries confidence and the evidence that produced it. Thin input stays uncertain.

The default path is deterministic, offline, and fast. There is no model and no network call unless you opt into the enhancer.

## Why this stack

Python and FastAPI. The compiler is a heuristic over punctuation, timing, repetition, and small lexicons. That work is string processing, and a typical message compiles in a few milliseconds, well under the 50 ms p95 budget, with no native extensions and no GPU. FastAPI gives a typed `POST /compile` contract and an OpenAPI document for free. The image is the official CPython slim base, which is published for both `linux/amd64` and `linux/arm64`.

Go or Rust would shave cold start. They would not make the detector easier to test, and the latency budget is not the constraint.

## Run it

Local:

```bash
python -m pip install -e ".[dev]"
python -m humansignal
```

The server binds `0.0.0.0:$PORT` (default port `8741`).

```bash
curl -s http://127.0.0.1:8741/health

curl -s http://127.0.0.1:8741/compile \
  -H 'content-type: application/json' \
  -d @examples/fine_after_friction.json
```

Interactive docs: [http://127.0.0.1:8741/docs](http://127.0.0.1:8741/docs).

Docker:

```bash
docker build -t humansignal .
docker run --rm -p 8741:8080 humansignal
```

Inside the container the service listens on `8080` (`PORT`). The mapping above publishes it on `8741`.

Multi-arch, still no GPU:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t humansignal .
```

`GET /health` returns `{"status":"ok","schema_version":"1.0.0"}`. `POST /compile` returns schema 1.0.0 JSON. `X-HumanSignal-Elapsed-Ms` is the server-side compile time.

## What you can send

One of `text` or `transcript`, not both.

```json
{ "text": "wait, I mean... what do you mean?", "language": "en", "speaker": "Sam" }
```

```json
{
  "transcript": [
    {"speaker": "Sam", "text": "um so the test fails when the token is empty?", "start": 0.0, "end": 3.4},
    {"speaker": "Rae", "text": "wait, I think... we do not handle the null path", "start": 5.2, "end": 9.0}
  ]
}
```

Timestamps are seconds. Gaps of about 0.85s or more become pause evidence. Overlap (the next turn starting before the previous one ends) is recorded too. Inline markers work alongside the audio: `[pause]`, `[long pause]`, `[laughter]`, `[overlap]`, plus fillers like `uh` / `um` and laughter tokens (`haha`, `jaja`, `www`, `555`, `哈哈`, `mdr`, `kkk`, `ㅋㅋ`, …).

`language` is optional (`en`, `es`, `ja`, or any other code). Omit it and HumanSignal sniffs script and function words. Unknown languages keep the structural detectors and skip the lexicon. That is deliberate degradation, not a failure.

### Turn by turn

Stateless, portable (send prior turns yourself):

```bash
curl -s http://127.0.0.1:8741/compile \
  -H 'content-type: application/json' \
  -d '{"language":"en","speaker":"Jordan","text":"Fine.","context":[{"speaker":"Alex","text":"You never listen. This is the third time."}]}'
```

Stateful, single process only:

```bash
curl -s http://127.0.0.1:8741/compile \
  -H 'content-type: application/json' \
  -d '{"session_id":"live-1","language":"en","speaker":"Alex","text":"You never listen. This is the third time."}'

curl -s http://127.0.0.1:8741/compile \
  -H 'content-type: application/json' \
  -d '{"session_id":"live-1","language":"en","speaker":"Jordan","text":"Fine."}'
```

`reset_session: true` clears that id first. The store is in-memory, capped, and dies with the process. Two replicas do not share it. Use `context` when the conversation window has to travel.

Zones are computed on the **current** input. Put the whole window in `transcript` when you want pressure and momentum spans for every turn. Put earlier turns in `context` (or a session) when you only want them as a baseline for drift, momentum, and a short reply that should inherit prior load. A clipped "Fine." is not treated as an emotional reset.

## Signal model

Structural features are always on. They do not need a language.

| Feature | What it sees | Feeds |
| --- | --- | --- |
| `!`, `?`, `?!`, `...` | bursts, questions, trailing off | excitement, confusion, frustration, pressure, momentum |
| ALL CAPS (acronyms like `API` ignored) | emphasis | excitement or frustration, depending on the company it keeps |
| elongation (`sooo`, `YESSS`) | vowel stretch, 4+ repeated letters, or a known stem like `yes`. A one-letter typo (`telll`, `helllo`) is not emphasis | excitement, sometimes frustration (`nooo`) |
| repetition (`no no no`, `I know. I know.`) | insistence or repair | frustration, excitement, momentum |
| laughter tokens and `[laughter]` | cross-language laughter | excitement, relief, connection |
| emoji and emoticons | a small classified set, plus a generic bucket | excitement, frustration, sarcasm/drift |
| fillers, hedges, self-corrections, `[pause]`, timestamp gaps | hesitation | confusion, pressure |
| currency amounts and commitment phrases (`$5K`, `paid`, `transaction`, `pago`, `支払い`) | a money or commitment ask, from the language pack plus a structural amount pattern | pressure, even with no hesitation. Bare `pay` / `sign` add weight but do not open a zone alone |
| speaking rate and turn length vs the other turns | speeding up, dragging, elaborating | momentum, pressure when the rate drops |
| lexical overlap, echo, steady gaps, short acknowledgements | people tracking each other | connection, and intent alignment when drift is absent |
| dismissive short replies (`Fine.`, `whatever`) after friction | words pulling away from the cues | intent drift, calm style. `ok` / `fine` / `vale` with no recent friction are backchannels, not shutdowns |

Language packs add phrases for frustration, excitement, confusion, relief, hedges, fillers, repairs, sarcasm, and agreement. They are data. English, Spanish, and Japanese ship as examples.

Scores are squashed with `1 - e^(-raw)` so extra cues matter less once a signal is already strong. They are not probabilities and they are not a partition: frustration and confusion can both be high. Confidence rises with tokens and with distinct evidence families, and it is capped at **0.8**. A one-word "hi" stays near the floor. The compiler would rather say "uncertain" than invent a mood.

### Output

Top-level keys, schema `1.0.0`:

| Key | Shape |
| --- | --- |
| `schema_version` | `"1.0.0"` |
| `emotional_state` | `frustration`, `excitement`, `confusion`, `relief` in 0..1, plus `dominant` (`neutral` when nothing clears the floor), `confidence`, `evidence` |
| `signal_strength` | how much paralinguistic material is present, not which emotion it is |
| `intent_alignment` | `score` (words agree with the other cues), `drift` (they diverge), `label` (`aligned`, `drifting`, `contradictory`, `uncertain`) |
| `pressure_zones` | spans where hesitation, fillers, pauses, hedges, deflection, or a commitment/money ask cluster |
| `momentum_zones` | spans where rate, length, exclamation, or other energy rise |
| `connection_score` | `score` plus `components`: `laughter`, `mirroring`, `rhythm`, `backchannel` |
| `recommended_response_style` | `{ "style", "confidence", "evidence" }`. `style` is one of `calm`, `hype`, `direct`, `playful`, `clarifying` |

`intent_alignment` is not "do these people agree?" A sincere argument is **aligned**: the angry words and the angry cues point the same way. "Fine." after that argument is **contradictory**: the word and the cues diverge.

`recommended_response_style` is a stance, not a script. Confusion leads to `clarifying`. Frustration or drift leads to `calm`. Shared laughter without a spike leads to `playful`. Excitement leads to `hype`. Otherwise the compiler says `direct`.

Evidence items look like `{ "feature", "detail", "weight", "span" }`. `weight` is relative trigger strength. `span.source` is `current` or `context`. Character offsets refer to the current turns joined by newlines.

The JSON Schema is `schema/compile-response.schema.json`. The OpenAPI document is `openapi.json` (and `GET /openapi.json` on the server).

## Examples

Full request/response bodies live in `examples/`. Numbers below are what the compiler emits for those files.

### Clipped reply after friction

Request `examples/fine_after_friction.json`:

```json
{
  "language": "en",
  "speaker": "Jordan",
  "text": "Fine.",
  "context": [
    {"speaker": "Alex", "text": "You never listen. This is the third time you ignored me."}
  ]
}
```

The current word is flat. The prior turn is not. Frustration is inherited (`context_carry`), drift is high, and the suggested stance is calm. Signal strength stays moderate because the current turn is one word; confidence on that strength stays modest.

```json
{
  "schema_version": "1.0.0",
  "emotional_state": {
    "frustration": 0.5198,
    "excitement": 0.0,
    "confusion": 0.0,
    "relief": 0.0,
    "dominant": "frustration",
    "confidence": 0.6169
  },
  "signal_strength": { "score": 0.288, "confidence": 0.3733 },
  "intent_alignment": {
    "score": 0.3012,
    "drift": 0.6988,
    "label": "contradictory",
    "confidence": 0.6195
  },
  "pressure_zones": [],
  "momentum_zones": [],
  "connection_score": { "score": 0.0, "confidence": 0.3 },
  "recommended_response_style": { "style": "calm", "confidence": 0.5799 }
}
```

Evidence on the drift: `clipped reply after prior frustration`. The full payload, including spans, is `examples/fine_after_friction.response.json`.

A bare `hi` (no context) is the other end: dominant `neutral`, emotion confidence about `0.14`, signal strength `0`, intent `uncertain`, style `direct`. Low signal, low confidence.

### Work conflict

`examples/work_conflict.json` — Alex has sent the deck, Jordan hedges through a 1.8s gap ("I was... in meetings, I guess."), Alex names the pattern ("third time", "you're always"), Jordan answers "Fine."

| | |
| --- | --- |
| dominant | frustration `0.69` (confusion `0.33` from the hedge and the pause) |
| intent | contradictory, drift `0.70` |
| pressure | `meetings` `0.70` — ellipsis, "I guess", the gap |
| momentum | none |
| connection | `0.40` — no laughter; some lexical overlap on "meetings" and a mild rhythm score. That is topical echo, not warmth |
| style | calm |

### Creative jam

`examples/creative_jam.json` — "EARLY", "YESSS", `[laughter]`, "haha", "??".

| | |
| --- | --- |
| dominant | excitement `0.62` (confusion `0.43` from "wait what" and the question cluster; both are real) |
| intent | aligned, drift `0` |
| momentum | `yesss drums absolutely wild` `0.63`, `chorus hits early drop` `0.54` |
| connection | `0.68`, laughter component `0.90` (both speakers) |
| style | hype |

### Coding pair

`examples/pair_coding.json` — "um", a 1.8s gap, "wait, I think...", "I mean", two questions, then a concrete trace.

| | |
| --- | --- |
| dominant | confusion `0.50` |
| intent | aligned (they are debugging, and the words match the hesitation) |
| pressure | `handle null path` `0.72` |
| connection | `0.27` |
| style | clarifying |

### Family

`examples/family.json` — "You never call", an apology with "um...", "I promise".

| | |
| --- | --- |
| dominant | frustration `0.53`, with relief `0.39` from the apology after the loaded turn |
| intent | aligned |
| pressure | `slammed sorry` `0.65` |
| style | calm |

## Language packs

Packs live in `src/humansignal/languages.py`. Register another at startup:

```python
from humansignal.languages import LanguagePack, register_pack

register_pack(LanguagePack(
    code="xx",
    name="Example",
    use_word_boundaries=True,  # False for scripts without spaces, like Japanese
    frustration_strong=("blargh",),
))
```

The real fields are `frustration_strong`, `frustration_weak`, `excitement_strong`, `excitement_weak`, `confusion`, `relief`, `positive`, `agreement`, `hedge_strong`, `hedge_weak`, `fillers`, `self_corrections`, `leading_self_corrections`, `deflection`, `dismissives`, `repair`, `sarcasm`, `backchannels`, `negation`, `commitment_strong`, `commitment_weak`, `loaded_repeats`, and `stopwords`. `loaded_repeats` (English `again`) only fires in a frustrated frame: `not`/`never` nearby, `yet again`, or emphatic punctuation (`again?!`). A neutral "look at it again" does not.

Word-boundary packs match case-insensitively. Japanese matches by substring, so the lists prefer longer phrases (`かもしれない`, not a bare `かも`) to avoid hitting pieces of unrelated words.

Passing `language` forces a pack. `language: "de"` with no German pack uses structural features only and lowers confidence slightly, because lexical cues are missing on purpose.

## Optional enhancer

Off by default. The compiler never opens a socket unless you set:

```bash
HUMAN_SIGNAL_ENHANCER=http
HUMAN_SIGNAL_ENHANCER_URL=http://127.0.0.1:9/enhance
HUMAN_SIGNAL_ENHANCER_TIMEOUT_S=0.8
```

The HTTP enhancer POSTs `{"request", "result"}` and must return a schema-valid result (or `{"result": ...}`). Timeouts and bad JSON fall back to the deterministic draft. Turning it on spends the latency budget; the 50 ms claim is for the default path.

The interface is `SignalEnhancer.enhance(request, result) -> CompileResponse` in `src/humansignal/enhancer.py`. `NoOpEnhancer` is the default.

## Develop

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m humansignal.bench
```

Tests cover each detector, English / Spanish / Japanese, unknown-language degradation, a pluggable pack, transcripts with timestamps, low-signal inputs, the HTTP contract, the JSON Schema, and latency. `python -m humansignal.bench` prints p50/p95/p99 for a multi-turn message and exits non-zero if p95 is 50 ms or above. On a laptop-class CPU this lands around 1–2 ms p95; the budget is there so a deploy stays honest.

## Limitations

- Heuristic, not clinical. Do not use the scores as a diagnosis, a risk score, or evidence about someone's character.
- Sarcasm and "Fine." are underdetermined. The compiler only fires drift when it can point at a cue (flat dismissive after heat, a hedge wrapped around a positive word, `/s`, a sarcasm emoji, quoted praise). Plenty of real sarcasm has none of those and will look aligned.
- Culture changes the cues. Caps, direct disagreement, silence, and laughter tokens (`www`, `555`, `kkk`, `mdr`) do not mean the same thing everywhere. The packs are small examples, not coverage.
- English function-word sniffing is a guess. Pass `language` when you know it.
- One speaker, or no timestamps, weakens connection and rate features. Confidence drops with them. The compiler will not invent a second person.
- Topic labels are content words in the span, not a topic model. "meetings" is a useful handle; it is not an ontology.
- Session memory is process-local. The filesystem on a platform like Render is ephemeral too; do not expect the session store to survive a restart.
- Confidence is an evidence function with a hard cap, not a calibrated probability from a held-out study.
- The enhancer is a hook, not a safety layer. If you turn a model on, you own that model's extra uncertainty.
