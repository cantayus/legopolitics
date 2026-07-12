from __future__ import annotations

import difflib
import uuid

from legopolitics.schemas import OCRRecord


def deduplicate_temporally(records: list[OCRRecord], threshold: float = 0.90) -> list[OCRRecord]:
    clusters: list[list[OCRRecord]] = []
    for record in sorted(records, key=lambda r: (r.frame_id, r.text_region_id)):
        placed = False
        for cluster in clusters:
            ratio = difflib.SequenceMatcher(
                None, cluster[-1].normalized_text.casefold(), record.normalized_text.casefold()
            ).ratio()
            if ratio >= threshold:
                cluster.append(record)
                placed = True
                break
        if not placed:
            clusters.append([record])
    output = []
    for cluster in clusters:
        first = cluster[0].model_copy(deep=True)
        first.temporal_cluster_id = f"ocrcluster_{uuid.uuid4().hex[:12]}"
        first.repetition_count = len(cluster)
        first.source_ids = [x.text_region_id for x in cluster]
        output.append(first)
    return output
