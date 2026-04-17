# PyPI Release Guide

This project is ready for local packaging and can be published once the GitHub URLs and PyPI credentials are set.

GitHub Actions are also included now:

- `CI` runs the test suite on Python `3.11`, `3.12`, and `3.13`
- `Package Check` builds the package and runs `twine check`
- `Publish to PyPI` publishes on GitHub release or manual dispatch

## Before The First Release

- replace `<your-username>` in `pyproject.toml` with the real GitHub username or org
- make sure the README reflects the public repo URL
- create a PyPI account if one does not exist
- create an API token on PyPI
- if using GitHub Actions publishing, configure PyPI trusted publishing for this repository

## Build The Package

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m build
```

This should create:

- `dist/*.tar.gz`
- `dist/*.whl`

## Check The Package Metadata

```bash
python -m twine check dist/*
```

## Upload To PyPI

```bash
python -m twine upload dist/*
```

If using an API token, the username is usually:

```text
__token__
```

## Good First Release Flow

1. bump version in `pyproject.toml`
2. run tests
3. build the package
4. run `twine check`
5. upload to PyPI
6. create a GitHub release with matching version notes
