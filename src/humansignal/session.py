"""Process-local session memory for turn-by-turn compiles.

This store is intentionally not durable. A container restart drops it, and
two replicas do not share it. Pass ``context`` when you need a stateless,
portable conversation window.
"""

from __future__ import annotations

import threading
import time

from humansignal.models import Segment


class SessionStore:
    """Bounded in-memory history keyed by ``session_id``."""

    def __init__(self, max_sessions: int = 512, max_turns: int = 80, ttl_s: float = 1800.0) -> None:
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, list[Segment]]] = {}

    def load(self, session_id: str) -> list[Segment]:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            found = self._items.get(session_id)
            if found is None:
                return []
            _, turns = found
            self._items[session_id] = (now, turns)
            return [turn.model_copy() for turn in turns]

    def append(self, session_id: str, turns: list[Segment]) -> None:
        if not turns:
            return
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            existing = self._items.get(session_id)
            history = list(existing[1]) if existing else []
            history.extend(turn.model_copy() for turn in turns)
            if len(history) > self.max_turns:
                history = history[-self.max_turns :]
            self._items[session_id] = (now, history)
            self._trim_sessions()

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)

    def _evict(self, now: float) -> None:
        expired = [key for key, (seen, _) in self._items.items() if now - seen > self.ttl_s]
        for key in expired:
            del self._items[key]

    def _trim_sessions(self) -> None:
        overflow = len(self._items) - self.max_sessions
        if overflow <= 0:
            return
        oldest = sorted(self._items.items(), key=lambda item: item[1][0])[:overflow]
        for key, _ in oldest:
            del self._items[key]
