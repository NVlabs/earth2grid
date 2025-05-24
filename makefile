sources = earth2grid

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

.PHONY: license
license:
	python tests/_license/header_check.py

format: license
	ruff check --fix $(sources) tests
	black $(sources) tests

lint: license
	pre-commit run --all-files

unittest:
	coverage run --source earth2grid/ -m pytest
	# requires vtk so don't run in ci
	# coverage run --source earth2grid/ -a -m pytest --doctest-modules earth2grid/ -vv

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
	# without setting the MPLBACKEND the build will stall on the pyvista
	MPLBACKEND=agg $(MAKE) -C docs html

push_docs: docs
	docs/push_docs.sh
