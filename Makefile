.PHONY: test install-dev

test:
	pytest -q

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

