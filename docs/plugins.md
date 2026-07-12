# Developing Plugins

Adapters use small protocols and registries. External packages can register entry points under:

- `legopolitics.detectors`
- `legopolitics.segmenters`
- `legopolitics.trackers`
- `legopolitics.vision_models`
- `legopolitics.ocr`
- `legopolitics.asr`
- `legopolitics.exporters`

A detector receives images, frame IDs, and a video ID and returns typed `DetectionBatch` objects. It must expose provenance and avoid hidden network calls.
