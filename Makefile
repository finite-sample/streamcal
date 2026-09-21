.PHONY: ci ci-docker evidence

ci:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright src
	uv run pydoclint src
	uv run pytest --cov
	uv run sphinx-build -W -b html docs docs/_build/html
	uv build

ci-docker:
	set -eu; for version in 3.12 3.14; do \
		docker run --rm -v "$(CURDIR):/work" -w /work \
			-e UV_PROJECT_ENVIRONMENT=/tmp/streamcal-venv \
			-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
			python:$$version-slim sh -c \
			"apt-get update && apt-get install -y --no-install-recommends git libatomic1 make && python -m pip install uv && uv sync --all-groups && make ci"; \
	done

evidence:
	uv run --group evidence python -m benchmarks.suite --output benchmarks/results/quality.json
	uv run --group evidence python -m benchmarks.resources --output benchmarks/results/resources.json
	uv run python -m benchmarks.render
