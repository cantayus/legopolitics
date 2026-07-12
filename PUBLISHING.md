# Publishing legopolitics on GitHub, GitHub Pages, TestPyPI, and PyPI

This guide assumes the repository owner is `cantayus` and the repository name is `legopolitics`. Replace those values if the repository will use a different owner.

## Prerequisites

Before public release:

1. Complete the institutional or qualified trademark review in `TRADEMARK_REVIEW.md`.
2. Confirm that the MIT `LICENSE` file is appropriate for project-authored code and review all third-party/model licenses separately.
3. Review `THIRD_PARTY_LICENSES.md` and `MODEL_LICENSES.md`.
4. Confirm that no model weights, credentials, copyrighted videos, output datasets, or private paths are committed.
5. Update the version in `pyproject.toml`, `src/legopolitics/version.py`, and `CITATION.cff` together.

The automated release workflows intentionally stop when the trademark review is incomplete or the license file is missing/provisional.

## 1. Create the GitHub repository

Create a new public repository named `legopolitics` under the `cantayus` account. Do not initialize it with a README, license, or `.gitignore`, because those files are already present locally.

From the repository root:

```bash
git init -b main
git add .
git commit -m "Initial public release of legopolitics"
git remote add origin https://github.com/cantayus/legopolitics.git
git push -u origin main
```

GitHub CLI alternative:

```bash
gh auth login
gh repo create cantayus/legopolitics \
  --public \
  --source=. \
  --remote=origin \
  --push
```

## 2. Configure the repository landing page

On the repository page, select the gear beside **About** and set:

- **Description:** `Multimodal analysis of political narratives in brick-built videos.`
- **Website:** `https://cantayus.github.io/legopolitics/`
- **Topics:** `python`, `computer-vision`, `multimodal`, `video-analysis`, `computational-social-science`, `political-communication`, `propaganda-analysis`, `narrative-analysis`, `huggingface`, `yolo`

Under **Settings → Social preview**, upload:

```text
docs/assets/legopolitics-social-preview.png
```

The image is 1280×640 and under 1 MB.

## 3. Enable GitHub Actions

Open the **Actions** tab and enable workflows if GitHub asks for confirmation. The repository contains:

- `ci.yml` — tests, linting, typing, docs, and package build
- `pages.yml` — MkDocs deployment to GitHub Pages
- `testpypi.yml` — manual TestPyPI publication
- `pypi.yml` — publication when a GitHub Release is published

Wait for the CI workflow to pass before publishing.

## 4. Enable GitHub Pages

Go to:

```text
Repository → Settings → Pages
```

Under **Build and deployment**, select:

```text
Source: GitHub Actions
```

Run the **Documentation** workflow manually or push a change to `main`. The site will be deployed to:

```text
https://cantayus.github.io/legopolitics/
```

## 5. Protect the main branch

Under **Settings → Branches** or **Settings → Rules → Rulesets**, create protection for `main`:

- Require a pull request before merging when collaborators are added.
- Require the CI status checks.
- Block force pushes.
- Block branch deletion.
- Require conversation resolution.

For a personal single-maintainer repository, direct pushes can remain allowed initially, but release workflows should remain restricted.

## 6. Create GitHub environments

Under **Settings → Environments**, create:

```text
testpypi
pypi
github-pages
```

For `pypi`, add a required reviewer if you want a manual approval before each production release. Do not store PyPI API tokens; the workflows use OpenID Connect Trusted Publishing.

## 7. Create PyPI and TestPyPI accounts

Create accounts on both PyPI and TestPyPI, verify the email address, and enable two-factor authentication.

The project name currently appears unregistered, but a pending Trusted Publisher does not reserve it. Configure the publishers and make the first release promptly.

## 8. Configure a pending Trusted Publisher on TestPyPI

In the TestPyPI account publishing settings, add a pending GitHub publisher with:

```text
PyPI project name: legopolitics
GitHub owner: cantayus
GitHub repository: legopolitics
Workflow filename: testpypi.yml
Environment name: testpypi
```

The workflow filename is the basename only, not `.github/workflows/testpypi.yml`.

## 9. Configure a pending Trusted Publisher on PyPI

In the PyPI account publishing settings, add a pending GitHub publisher with:

```text
PyPI project name: legopolitics
GitHub owner: cantayus
GitHub repository: legopolitics
Workflow filename: pypi.yml
Environment name: pypi
```

No PyPI password or API token is placed in GitHub.

## 10. Complete the release gates

Update `TRADEMARK_REVIEW.md` only after the review has genuinely been completed:

```yaml
trademark_review:
  completed: true
  reviewer: "Name or office"
  review_date: "YYYY-MM-DD"
  approved_public_name: "legopolitics"
  notes: "Summary of the written review"
```

The repository now contains the standard MIT License. Preserve `LICENSE` and `NOTICE` in redistributed copies, and separately confirm all third-party and model-asset terms.

Run locally:

```bash
python scripts/check_release_readiness.py
```

## 11. Build and verify locally

Use a clean environment:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff format --check src tests
ruff check src tests
mypy src/legopolitics
python -m pytest
mkdocs build --strict
rm -rf dist build
python -m build
python -m twine check --strict dist/*
```

Windows PowerShell equivalents for removing build folders:

```powershell
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
```

## 12. Publish to TestPyPI

Commit and push the release-ready files. Then:

```text
GitHub repository → Actions → TestPyPI → Run workflow
```

When the workflow succeeds, install from TestPyPI in a clean environment:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  legopolitics==0.1.0
```

The extra PyPI index is necessary because project dependencies may not exist on TestPyPI.

Test:

```bash
legopolitics version
legopolitics doctor
legopolitics benchmark --seconds 1
```

## 13. Create the production release

Ensure the version is unique and committed. Create and push a signed or annotated tag:

```bash
git tag -a v0.1.0 -m "legopolitics 0.1.0"
git push origin v0.1.0
```

On GitHub:

```text
Releases → Draft a new release
```

- Choose tag `v0.1.0`.
- Set title `legopolitics 0.1.0`.
- Summarize major capabilities, installation, known limitations, and licensing notices.
- Publish the release.

Publishing the GitHub Release triggers `pypi.yml`. The workflow builds the wheel and source distribution, validates them, and publishes through Trusted Publishing.

## 14. Verify PyPI

In a clean environment:

```bash
python -m pip install legopolitics==0.1.0
legopolitics version
legopolitics doctor
```

Confirm that the PyPI page shows:

- Logo and README
- General description
- Capabilities
- Installation examples
- Project links
- Python-version classifiers
- Trademark disclaimer
- Release files
- Digital attestations generated by Trusted Publishing

## 15. Future releases

For every release:

1. Update `CHANGELOG.md`.
2. Increment the version in all version files.
3. Run CI and local build checks.
4. Publish the same version to TestPyPI.
5. Install and test from TestPyPI.
6. Tag the commit.
7. Publish a GitHub Release.
8. Verify the PyPI installation.

PyPI does not allow replacing an already uploaded file. A failed or incorrect release requires a new version number.
