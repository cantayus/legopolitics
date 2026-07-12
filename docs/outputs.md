# Outputs

Each video output contains:

- `result.json`: complete typed result.
- `run_manifest.sqlite`: run state, stage state, errors, locks, artifacts, and data tables.
- `video_summary.xlsx`: researcher-facing workbook with a disclaimer and data dictionary.
- `report.html`: local report.
- `tables/*.parquet`: columnar tables when `pyarrow` is installed.
- `jsonl/*.jsonl`: raw and parsed records.
- sampled frames, crops, masks, audio, caches, validation data, and logs.

Excel cells are sanitized against formula injection, files are written atomically, and the workbook is reopened after creation as a validation step.
