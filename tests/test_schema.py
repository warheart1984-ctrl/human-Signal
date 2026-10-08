"""Responses validate against the published JSON Schema."""

import json

from jsonschema import Draft202012Validator

from helpers import SCHEMA_PATH, TOP_LEVEL, load_example
from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest


def test_examples_match_schema_and_top_level_keys() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    names = (
        "work_conflict.json",
        "creative_jam.json",
        "pair_coding.json",
        "family.json",
    )
    for name in names:
        result = load_example(name)
        payload = result.model_dump(exclude_none=True)
        assert set(payload) == TOP_LEVEL
        errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
        assert errors == [], errors


def test_low_signal_payload_matches_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    payload = compile_signals(CompileRequest(text="hi", language="en")).model_dump(exclude_none=True)
    assert sorted(validator.iter_errors(payload), key=lambda err: list(err.path)) == []
