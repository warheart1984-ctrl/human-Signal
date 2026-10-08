"""p95 for a typical multi-turn message stays under 50 ms."""

from humansignal.bench import run_benchmark


def test_typical_message_p95_under_50ms() -> None:
    stats = run_benchmark(iterations=80, warmup=10)
    print(
        "benchmark "
        f"p50={stats['p50_ms']:.2f}ms "
        f"p95={stats['p95_ms']:.2f}ms "
        f"p99={stats['p99_ms']:.2f}ms "
        f"mean={stats['mean_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms"
    )
    assert stats["p95_ms"] < 50.0
