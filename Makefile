.PHONY: install test bench run

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

bench:
	python -m humansignal.bench

run:
	python -m humansignal
