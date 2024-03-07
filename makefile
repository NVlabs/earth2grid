sources = earth2grid

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests
	mypy $(sources) tests

unittest:
	coverage run --source earth2grid/ -m pytest
	coverage run --source earth2grid/ -a -m pytest --doctest-modules earth2grid/ -vv

coverage:
	coverage report

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage


.PHONY: docs
docs:
	$(MAKE) -C docs html

push_docs: docs
	docs/push_docs.sh
