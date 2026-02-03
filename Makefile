PYTHON ?= python

.PHONY: clean test

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
