PY ?= python
PKG = vinexplainnet

.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "targets: install dev lint type test smoke docker clean"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

dev:
	$(PY) -m pip install -e ".[dev,export]"

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .
	$(PY) -m isort --check-only .

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest

smoke:
	$(PY) -m $(PKG).studio train --treatment configs/experiment/_smoke.conf
	$(PY) -m $(PKG).studio evaluate --treatment configs/experiment/_smoke.conf

docker:
	docker build -t $(PKG):latest .

clean:
	rm -rf takes .mypy_cache .ruff_cache .pytest_cache *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
