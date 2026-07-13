from __future__ import annotations

import sqlite3
from pathlib import Path

from legopolitics.schemas import ManualCorrection


def append_correction(database: Path, correction: ManualCorrection) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO manual_corrections VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                correction.correction_id,
                correction.video_id,
                correction.unit_type,
                correction.unit_id,
                correction.action,
                str(correction.original_value),
                str(correction.corrected_value),
                correction.coder_id,
                correction.notes,
                correction.created_at.isoformat(),
            ),
        )
