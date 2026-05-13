PYTHON ?= python

.PHONY: clean build check compile release-help

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

compile:
	$(PYTHON) -m compileall fast_exp_analytics

build: clean
	$(PYTHON) -m build

check:
	$(PYTHON) -m twine check dist/*

release-help:
	@echo "1) bump version"
	@echo "2) update CHANGELOG.md"
	@echo "3) make build"
	@echo "4) make check"
