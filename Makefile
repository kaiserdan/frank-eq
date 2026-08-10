.PHONY: install test validate smoke full clean

install:
	python -m pip install -e '.[dev]'

test:
	python -m compileall -q src scripts
	pytest -q

validate:
	python scripts/validate_repo.py

smoke:
	frank-eq run-stage0 --config configs/stage0/synthetic_smoke.yaml --out runs/synthetic-smoke

full:
	frank-eq run-stage0 --config configs/stage0/synthetic_full.yaml --out runs/synthetic-stage0-v1

clean:
	rm -rf runs .pytest_cache .ruff_cache
