# Configuration

Configuration uses versioned YAML validated by Pydantic. See `configs/minimal.yaml` and `configs/comprehensive.yaml`. Environment overrides use double underscores, for example:

```bash
LEGOPOLITICS__SAMPLING__FPS=2.0 legopolitics analyze-video ...
```

The effective configuration and its SHA-256 fingerprint are written to every output directory. Configuration changes invalidate the completed-result resume check and stage caches.
