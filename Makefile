# Commands used by CI

.PHONY: format lint test

format:
	ruff check --fix
	ruff format

lint:
	ruff check
	ruff format --diff
	mypy src/sarkit_processing

test:
	pytest -s
