"""The checked-in OpenAPI document matches the running app."""

import json
from pathlib import Path

from humansignal.main import app

ROOT = Path(__file__).resolve().parents[1]


def test_openapi_covers_health_and_compile() -> None:
    spec = json.loads((ROOT / "openapi.json").read_text(encoding="utf-8"))
    live = app.openapi()
    assert set(spec["paths"]) == set(live["paths"])
    assert {"/health", "/compile"} <= set(spec["paths"])
    assert spec["info"]["version"] == "1.0.0"
    assert "post" in spec["paths"]["/compile"]
    assert "get" in spec["paths"]["/health"]
