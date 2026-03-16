.PHONY: test lint format coverage clean install dev

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest aviris_tools/tests/ -v --tb=short

coverage:
	pytest --cov=aviris_tools --cov=hsi_toolkit --cov-report=term-missing

lint:
	flake8 aviris_tools/ hsi_toolkit/

format:
	black aviris_tools/ hsi_toolkit/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
