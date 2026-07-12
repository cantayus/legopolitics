from __future__ import annotations

import json
import uuid
from pathlib import Path

from legopolitics.constants import DISCLAIMER
from legopolitics.review.forms import REVIEW_ACTIONS
from legopolitics.review.state import load_validation_records


def run_app(validation_file: str, corrections_file: str) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install legopolitics[review] to use the review application") from exc
    st.title("legopolitics validation review")
    st.caption(DISCLAIMER)
    records = load_validation_records(Path(validation_file))
    if not records:
        st.info("No validation records found.")
        return
    index = st.number_input("Record", 0, len(records) - 1, 0)
    record = records[int(index)]
    st.json(record)
    action = st.selectbox("Action", REVIEW_ACTIONS)
    label = st.text_input("Corrected label", value=str(record.get("suggested_label") or ""))
    coder = st.text_input("Coder ID")
    notes = st.text_area("Notes")
    if st.button("Save correction"):
        correction = {
            "correction_id": f"correction_{uuid.uuid4().hex[:16]}",
            "video_id": record.get("video_id"),
            "unit_type": record.get("unit_type"),
            "unit_id": record.get("unit_id"),
            "action": action,
            "corrected_value": label,
            "coder_id": coder,
            "notes": notes,
        }
        target = Path(corrections_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(correction, ensure_ascii=False) + "\n")
        st.success("Correction saved without overwriting the automated record.")
