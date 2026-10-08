"""Latency check for a typical multi-turn message.

Run ``python -m humansignal.bench``. Exits non-zero if p95 is at or above 50 ms.
The timed path is in-process ``compile_signals``: no network, no model.
"""

from __future__ import annotations

import statistics
import time

from humansignal.compiler import compile_signals
from humansignal.models import CompileRequest, Segment

BUDGET_MS = 50.0

TYPICAL = CompileRequest(
    language="en",
    transcript=[
        Segment(
            speaker="Alex",
            text="I sent the deck yesterday and you still haven't looked at it.",
            start=0.0,
            end=3.6,
        ),
        Segment(
            speaker="Jordan",
            text="I was... in meetings, I guess.",
            start=5.4,
            end=8.2,
        ),
        Segment(
            speaker="Alex",
            text="You're always in meetings. This is the third time.",
            start=8.5,
            end=11.4,
        ),
        Segment(speaker="Jordan", text="Fine.", start=12.0, end=12.6),
        Segment(
            speaker="Alex",
            text="Can you just open it tonight? The client review is at 9 and I am not doing this again.",
            start=13.0,
            end=18.5,
        ),
    ],
)


def run_benchmark(iterations: int = 300, warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        compile_signals(TYPICAL)
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        compile_signals(TYPICAL)
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()

    def percentile(rank: float) -> float:
        if not samples:
            return 0.0
        index = min(len(samples) - 1, max(0, int(round(rank * (len(samples) - 1)))))
        return samples[index]

    return {
        "iterations": float(iterations),
        "warmup": float(warmup),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "mean_ms": statistics.fmean(samples),
        "max_ms": samples[-1],
        "budget_ms": BUDGET_MS,
    }


def main() -> None:
    stats = run_benchmark()
    for key in ("iterations", "warmup", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms", "budget_ms"):
        value = stats[key]
        if key in {"iterations", "warmup"}:
            print(f"{key}={int(value)}")
        else:
            print(f"{key}={value:.3f}")
    if stats["p95_ms"] >= BUDGET_MS:
        raise SystemExit(f"p95 {stats['p95_ms']:.2f} ms exceeds {BUDGET_MS:.0f} ms")


if __name__ == "__main__":
    main()
