"""Optional post-pass. Off unless ``HUMAN_SIGNAL_ENHANCER=http``.

The default compiler never calls the network. An enhancer may adjust a
schema-valid result, but a failure returns the deterministic draft unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Protocol

from humansignal.models import CompileRequest, CompileResponse

logger = logging.getLogger("humansignal.enhancer")


class SignalEnhancer(Protocol):
    """Post-processor. Implementations must tolerate being skipped entirely."""

    name: str

    def enhance(self, request: CompileRequest, result: CompileResponse) -> CompileResponse:
        """Return a schema-valid result. Returning ``result`` is always acceptable."""


class NoOpEnhancer:
    """The default. Returns the heuristic result with no extra work."""

    name = "none"

    def enhance(self, request: CompileRequest, result: CompileResponse) -> CompileResponse:
        return result


class HttpEnhancer:
    """POST the draft JSON to an external URL. Not used unless configured.

    Timeouts and bad payloads fall back to the draft so a flaky enhancer
    cannot take the compile path down. Enabling this voids the offline
    latency budget.
    """

    name = "http"

    def __init__(self, url: str, timeout_s: float = 0.8) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def enhance(self, request: CompileRequest, result: CompileResponse) -> CompileResponse:
        payload = json.dumps(
            {"request": request.model_dump(mode="json"), "result": result.model_dump(mode="json")}
        ).encode("utf-8")
        http_request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                body = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("enhancer unreachable, using draft: %s", exc)
            return result
        try:
            parsed = json.loads(body)
            updated = parsed["result"] if isinstance(parsed, dict) and "result" in parsed else parsed
            return CompileResponse.model_validate(updated)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("enhancer returned an unusable payload, using draft: %s", exc)
            return result


def load_enhancer() -> SignalEnhancer:
    """Build the enhancer named by the environment. Default is :class:`NoOpEnhancer`."""
    kind = os.environ.get("HUMAN_SIGNAL_ENHANCER", "none").strip().lower()
    if kind in {"", "none", "off", "noop"}:
        return NoOpEnhancer()
    if kind == "http":
        url = os.environ.get("HUMAN_SIGNAL_ENHANCER_URL", "").strip()
        if not url:
            raise RuntimeError("HUMAN_SIGNAL_ENHANCER=http requires HUMAN_SIGNAL_ENHANCER_URL")
        timeout = float(os.environ.get("HUMAN_SIGNAL_ENHANCER_TIMEOUT_S", "0.8"))
        return HttpEnhancer(url, timeout_s=timeout)
    raise RuntimeError(f"unknown HUMAN_SIGNAL_ENHANCER={kind!r} (expected none or http)")
