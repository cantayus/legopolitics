from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from legopolitics.constants import DISCLAIMER
from legopolitics.storage.atomic import atomic_write_text

TEMPLATE = Template("""<!doctype html><html><head><meta charset='utf-8'><title>legopolitics report</title>
<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.4rem;text-align:left}.notice{background:#f5f5f5;padding:1rem}code{word-break:break-all}</style></head><body>
<h1>legopolitics analysis report</h1><div class='notice'>{{ disclaimer }}</div>
<h2>Video</h2><table>{% for k,v in video.items() %}<tr><th>{{k}}</th><td>{{v}}</td></tr>{% endfor %}</table>
<h2>Processing summary</h2><table>{% for k,v in summary.items() %}<tr><th>{{k}}</th><td>{{v}}</td></tr>{% endfor %}</table>
<h2>Narrative segments</h2><table><tr><th>Label</th><th>Start</th><th>End</th><th>Confidence</th></tr>{% for n in narratives %}<tr><td>{{n.label}}</td><td>{{n.start_timestamp}}</td><td>{{n.end_timestamp}}</td><td>{{n.confidence}}</td></tr>{% endfor %}</table>
<h2>Errors</h2><table><tr><th>Stage</th><th>Type</th><th>Message</th></tr>{% for e in errors %}<tr><td>{{e.stage}}</td><td>{{e.exception_type}}</td><td>{{e.message}}</td></tr>{% endfor %}</table>
<h2>Provenance</h2><pre>{{ provenance }}</pre><footer><p>{{ disclaimer }}</p></footer></body></html>""")


def write_html_report(path: Path, result: Any) -> Path:
    summary = {
        "frames": len(result.frames),
        "detections": len(result.detections),
        "tracks": len(result.tracks),
        "ocr_regions": len(result.ocr),
        "transcript_segments": len(result.transcript),
        "relations": len(result.relations),
        "errors": len(result.errors),
    }
    html = TEMPLATE.render(
        disclaimer=DISCLAIMER,
        video=result.video.model_dump(mode="json"),
        summary=summary,
        narratives=result.narrative_segments,
        errors=result.errors,
        provenance=result.provenance.model_dump_json(indent=2)
        if result.provenance
        else "Unavailable",
    )
    return atomic_write_text(path, html)
