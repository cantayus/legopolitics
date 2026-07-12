# Third-Party Dependency Notices

The original `legopolitics` package source code and project-authored
documentation are licensed under the MIT License. Third-party software and
external assets are not relicensed by that license.

Important optional components include:

- **Ultralytics**: optional YOLO adapter. Ultralytics currently offers
  AGPL-3.0 and Enterprise licensing options. Review the terms that apply to
  your use and redistribution.
- **PyTorch and Transformers**: used by several optional local-model adapters;
  each package remains subject to its own license.
- **FFmpeg**: invoked as an external executable when available and subject to
  the license of the installed build.
- **pyannote.audio**: optional diarization software; package and model terms
  may differ.
- **Streamlit**: optional local review interface.
- **Hugging Face models**: model repositories can use different licenses,
  access restrictions, and acceptable-use terms. Review each model card.

## Model checkpoint in the source repository

The source repository may contain the user-supplied checkpoint:

```text
model_assets/lego_figure_yolo26s/best.pt
```

That checkpoint is not included in the PyPI wheel and is not automatically
covered by the package's MIT License. Redistribute it only after confirming the
training-data rights, checkpoint ownership, and all applicable Ultralytics and
model terms. Its model card and metadata must accompany any authorized
redistribution.

This notice is informational and is not legal advice.
