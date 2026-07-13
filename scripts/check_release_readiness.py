"""Fail a public release when legal review or package metadata is incomplete."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
override = os.getenv("LEGOPOLITICS_DEV_RELEASE_OVERRIDE", "").lower() in {
    "1",
    "true",
    "yes",
}

errors: list[str] = []

review_path = ROOT / "TRADEMARK_REVIEW.md"
review = review_path.read_text(encoding="utf-8") if review_path.exists() else ""
if not re.search(r"completed:\s*true", review, re.IGNORECASE):
    errors.append("Trademark review is incomplete in TRADEMARK_REVIEW.md.")
if not re.search(r"approved_public_name:\s*[\"']?legopolitics", review, re.IGNORECASE):
    errors.append("TRADEMARK_REVIEW.md does not record legopolitics as the approved public name.")

license_path = ROOT / "LICENSE"
license_text = license_path.read_text(encoding="utf-8") if license_path.exists() else ""
license_pending_markers = (
    "license selection is intentionally pending",
    "replace this file with an approved",
    "private research and evaluation while",
)
if not license_text.strip():
    errors.append("LICENSE is missing or empty.")
elif any(marker in license_text.lower() for marker in license_pending_markers):
    errors.append("LICENSE is still the provisional pre-release notice.")

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
if not re.search(r'^name\s*=\s*["\']legopolitics["\']', pyproject, re.MULTILINE):
    errors.append("pyproject.toml project name is not legopolitics.")

if errors and not override:
    print("Public release blocked:", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)
    raise SystemExit(2)

if errors:
    print("Release readiness checks bypassed by LEGOPOLITICS_DEV_RELEASE_OVERRIDE:")
    for error in errors:
        print(f"- {error}")
else:
    print("Release readiness checks passed.")
