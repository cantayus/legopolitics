# Validation Report

Validation date: 2026-07-12

## Successful checks

- Ruff linting: passed.
- Mypy static typing: passed for 166 source modules.
- Pytest: 32 tests passed with `ResourceWarning` treated as an error.
- Coverage: 66.86%; CI threshold is 60%.
- MkDocs: strict documentation build passed.
- Build: wheel and source distribution built successfully.
- Twine metadata check: wheel and source distribution passed.
- Fresh virtual-environment installation: passed.
- `pip check`: no broken requirements.
- Installed CLI: `version`, `about`, `--help`, `doctor`, `benchmark`, `analyze-video`, and `validate` executed successfully.
- End-to-end installed-wheel run: Excel, Parquet, JSONL, SQLite, HTML, environment, provenance, and YAML configuration artifacts were created.
- Excel workbook reopened successfully with 24 expected sheets.
- Parquet integration produced 21 tables with PyArrow installed.
- Public-release trademark gate blocked an incomplete review and passed only with the explicit development override.
- Source distribution includes documentation, configurations, notebooks, tests, scripts, Docker files, and workflows.

## Exact final validation commands

```bash
ruff check src tests
mypy src/legopolitics
PYTHONPATH=src pytest -q -W error::ResourceWarning
PYTHONPATH=src pytest --cov=legopolitics --cov-report=term --cov-fail-under=60 -q
mkdocs build --strict --clean
python -m build
twine check dist/*
pip install --force-reinstall dist/legopolitics-0.1.0-py3-none-any.whl
pip check
legopolitics version
legopolitics about
legopolitics --help
legopolitics doctor --output .
legopolitics benchmark --output /tmp/legopolitics-final-benchmark --seconds 1
legopolitics analyze-video --video synthetic.mp4 --config configs/minimal.yaml --output /tmp/legopolitics-final-output
legopolitics validate --output /tmp/legopolitics-final-output
python scripts/check_trademark_review.py
LEGOPOLITICS_DEV_RELEASE_OVERRIDE=1 python scripts/check_trademark_review.py
```

The first trademark-gate command intentionally exited with status 2 because the review is incomplete.

## Environment exercised

The final wheel was installed and exercised on Linux with Python 3.13.5 in CPU mode. The package metadata and CI matrix target Python 3.10, 3.11, and 3.12; those CI jobs are configured but were not executed inside this artifact-building environment.
