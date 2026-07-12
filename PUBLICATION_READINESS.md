# Publication Readiness

## Implemented for GitHub and PyPI

- Expanded repository README with package overview, capability matrix, installation, CLI and Python examples, model configuration, outputs, batch processing, documentation, citation, and legal notices.
- Added optimized repository logo, favicon, and 1280×640 social-preview image.
- Added a complete end-to-end tutorial covering sampling, custom YOLO weights, open-vocabulary detection, tracking, segmentation, Florence-2/CLIP/DINOv2, OCR, audio, research context, codebooks, relationships, narrative segmentation, batches, and validation.
- Added a detailed publication guide for GitHub, GitHub Pages, TestPyPI, PyPI Trusted Publishing, and future releases.
- Added MkDocs Material configuration and automatic GitHub Pages deployment.
- Added modern CI and Trusted Publishing workflows with separate `testpypi` and `pypi` GitHub environments.
- Added project links to package metadata so PyPI can display the homepage, documentation, repository, issue tracker, and changelog.
- Added a release-readiness script that blocks publication while trademark review is incomplete or the license file is missing/provisional.

## Validation performed on July 12, 2026

- `ruff format --check src tests`: passed.
- `ruff check src tests`: passed.
- `python -m pytest -vv`: 32 tests passed.
- `mkdocs build --strict`: passed.
- `python -m build`: wheel and source distribution built successfully.
- `python -m twine check --strict dist/*`: both distributions passed.
- PyPI README rendering validation: passed through Twine.
- Release-readiness gate: accepts the MIT License and continues to block release while the trademark review remains incomplete.

## Actions still required from the project owner

1. Complete a genuine institutional or qualified trademark review and update `TRADEMARK_REVIEW.md`.
2. The project-authored code is now licensed under MIT; confirm third-party and model-asset obligations before redistribution.
3. Create the GitHub repository and push the source.
4. Enable GitHub Pages with **Source: GitHub Actions**.
5. Create and secure PyPI and TestPyPI accounts.
6. Configure pending Trusted Publishers using the exact workflow and environment names in `PUBLISHING.md`.
7. Run the TestPyPI workflow and test installation.
8. Publish a GitHub Release to trigger the production PyPI workflow.
