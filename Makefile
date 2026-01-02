.PHONY: test install-dev

test:
	pytest -q

install-dev:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
