# Local model weights

Place private or locally trained checkpoints in this directory. Model files are
not included in source distributions and should normally not be committed to
GitHub.

Recommended filename for the LEGO-figure detector:

```text
models/lego_figure_best.pt
```

After copying the real checkpoint, inspect it with:

```bash
legopolitics inspect-yolo-model --weights models/lego_figure_best.pt
```

A valid YOLO detection checkpoint should report a nonzero file size, a task such
as `detect`, and a class map. If it reports `classify`, it cannot localize figures
inside complete frames and a detection checkpoint is required instead.
