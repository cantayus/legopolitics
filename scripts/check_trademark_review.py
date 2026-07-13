import os
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "TRADEMARK_REVIEW.md")
override = os.getenv("LEGOPOLITICS_DEV_RELEASE_OVERRIDE", "").lower() in {"1", "true", "yes"}
text = path.read_text(encoding="utf-8") if path.exists() else ""
completed = bool(re.search(r"completed:\s*true", text, re.IGNORECASE))
if not completed and not override:
    print("Public release blocked: trademark review is incomplete.", file=sys.stderr)
    raise SystemExit(2)
print("Trademark release gate passed" + (" by development override" if override and not completed else ""))
