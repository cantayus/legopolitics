# Troubleshooting

Run:

```bash
legopolitics doctor
```

This reports Python, FFmpeg, PyTorch, CUDA, GPU, optional libraries, token presence without values, write access, and trademark-review state.

For CUDA out-of-memory errors, reduce inference batch size, enable low-VRAM mode, reduce model/image size, or unload models between stages. Failed optional stages are recorded in SQLite and JSONL while successful stages remain available.
