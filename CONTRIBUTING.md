# Contributing

Use Python 3.12+ and a C++17 compiler. From a clean checkout:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest --cov=vector_server --cov-fail-under=90
mypy --strict src
ruff check src tests setup.py
black --check src tests setup.py
```

Native code is in `native/`; the typed engine and API are in
`src/vector_server/`. Keep the existing Python namespace for compatibility.

For geometry changes, add finite-difference or invariant tests. For API changes,
exercise the request and error behavior. Compare SVG previews before changing
photo or curve heuristics; smaller pixel error alone does not establish better
perceptual quality.

Open an issue with a shareable input image when reporting quality problems.
Pull requests should explain the change and include relevant validation.

Releases use semantic version tags. Publishing a release triggers native
AMD64 and ARM64 Docker builds, smoke tests, and downloadable image archives.
