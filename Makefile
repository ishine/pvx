.PHONY: install install-dev lint format typecheck test coverage docs pre-commit bench depsync

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

install-dev:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e ".[dev]"
	python3 -m pip install pre-commit

lint:
	ruff check src/pvx src/pvxalgorithms scripts tests

format:
	ruff format src/pvx src/pvxalgorithms scripts tests

typecheck:
	mypy src/pvx/core/attribution.py scripts/scripts_apply_attribution.py

depsync:
	python3 scripts/scripts_check_dependency_sync.py

test:
	python3 -m unittest discover -s tests -p "test_*.py"

coverage:
	coverage run -m unittest discover -s tests -p "test_*.py"
	coverage report --fail-under=45

docs:
	python3 scripts/scripts_generate_python_docs.py
	python3 scripts/scripts_generate_theory_docs.py
	python3 scripts/scripts_generate_docs_extras.py
	python3 scripts/scripts_generate_html_docs.py

pre-commit:
	pre-commit run --all-files

bench:
	python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --strict-corpus --determinism-runs 2
