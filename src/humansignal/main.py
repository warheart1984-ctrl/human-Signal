"""HTTP surface: ``GET /health`` and ``POST /compile``."""

from __future__ import annotations

import logging
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Response

from humansignal.compiler import compile_signals
from humansignal.document import current_segments
from humansignal.enhancer import SignalEnhancer, load_enhancer
from humansignal.errors import InputError
from humansignal.models import CompileRequest, CompileResponse, HealthResponse
from humansignal.session import SessionStore
from humansignal.version import SCHEMA_VERSION

logger = logging.getLogger("humansignal")


def create_app(
    enhancer: SignalEnhancer | None = None,
    sessions: SessionStore | None = None,
) -> FastAPI:
    app = FastAPI(
        title="HumanSignal",
        version=SCHEMA_VERSION,
        summary="Emotional compiler for conversational signals.",
        description=(
            "Turns messy conversational text or a timed transcript into structured, "
            "explained signal JSON. Heuristic and deterministic. Not therapy, not a "
            "clinical instrument, and not a general sentiment model. The optional "
            "enhancer is off unless HUMAN_SIGNAL_ENHANCER is set."
        ),
    )
    store = sessions or SessionStore()
    active = enhancer if enhancer is not None else load_enhancer()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", schema_version="1.0.0")

    @app.post("/compile", response_model=CompileResponse, response_model_exclude_none=True)
    def compile_endpoint(body: CompileRequest, response: Response) -> CompileResponse:
        started = time.perf_counter()
        try:
            prior = []
            if body.session_id:
                if body.reset_session:
                    store.reset(body.session_id)
                else:
                    prior = store.load(body.session_id)
            result = compile_signals(body, prior=prior)
            result = active.enhance(body, result)
            if body.session_id:
                store.append(body.session_id, current_segments(body))
        except InputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-HumanSignal-Elapsed-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-HumanSignal-Language"] = body.language or "auto"
        return result

    return app


app = create_app()


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8741"))
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    uvicorn.run(app, host=host, port=port)
