.PHONY: test install-dev

test:
	pytest -q

smoke:
	python -m analysis.chem.pipeline --tree_input tests/data/hetero_tree_min.json --k 10 --seed 0 --timestamp 2026-01-02T00:00:00+00:00 --select_k 5 --selection maxmin --out smoke_pipeline.json
	python -m analysis.chem.report --input smoke_pipeline.json --out_dir . --stem smoke

install-dev:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
