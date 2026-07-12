# YOLO26s Integration Report

## Supplied checkpoint

- File: `model_assets/lego_figure_yolo26s/best.pt`
- Size: 20,312,709 bytes
- SHA-256: `b027885bd139f7eced7e39beb4e734ad7fa9a88ef28c77cecea75d17bdc5e98b`
- Task: object detection
- Architecture: YOLO26s
- Classes: `human` (0), `lego_figure` (1)
- Training image size: 640
- Epochs: 50

## Final supplied validation metrics

- Precision: 0.94639
- Recall: 0.70621
- mAP50: 0.79407
- mAP50–95: 0.59977

## Package changes

- Version increased to 0.1.1.
- YOLO class filters now accept class names or IDs.
- `lego_figure` is the default target in `configs/lego_figure_yolo26s.yaml`.
- Unknown class names and IDs fail with an informative error.
- Model provenance now records class maps and filters.
- The checkpoint and selected training artifacts are stored outside `src/`.
- The checkpoint is excluded from the wheel but included in this source-repository bundle.
- Git LFS attributes were added for `.pt`, `.onnx`, and `.engine` assets.
- A model card and machine-readable metadata file were added.

## Real inference smoke test

The model was loaded with Ultralytics on CPU and run on the supplied `val_batch0_pred.jpg` image using:

- Image size: 640
- Confidence threshold: 0.25
- Class filter: `lego_figure`

Result:

- 10 detections
- All returned detections were `lego_figure`
- Maximum confidence: 0.80405

Additional supplied validation images produced 12 and 13 LEGO-figure detections, with maximum confidence reaching 0.96171.

## Validation

- 42 automated tests passed.
- Ruff formatting passed.
- Ruff linting passed.
- Strict MkDocs build passed.
- Wheel and source distribution built successfully.
- Strict Twine checks passed.
- Full mypy validation was attempted but exceeded the available execution window; no claim is made that this final revision passed full mypy.

## Recommended command

```bash
python -m pip install -e ".[yolo,huggingface]"

legopolitics analyze-video \
  --video "videos/example.mp4" \
  --config "configs/lego_figure_yolo26s.yaml" \
  --output "outputs/example"
```

Review the dataset, checkpoint, and Ultralytics licensing before publicly redistributing the model weights.
