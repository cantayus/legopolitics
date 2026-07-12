# Integrating the custom LEGO-figure YOLO model

The YOLO checkpoint is an external research asset. It is loaded at runtime and
is not embedded in the Python wheel or source distribution.

## The current uploaded ZIP cannot be used

The supplied `yolo26s_results.zip` contains the expected training-result names,
including `weights/best.pt`, but the actual files in the ZIP have length zero.
Only macOS `__MACOSX/._*` metadata entries contain bytes. Re-create the archive
after confirming that `best.pt` has a realistic nonzero size.

On macOS or Linux:

```bash
ls -lh /path/to/yolo26s_results/weights/best.pt
zip -r yolo26s_results_complete.zip yolo26s_results -x "*/__MACOSX/*" "*/.DS_Store"
unzip -l yolo26s_results_complete.zip | grep best.pt
```

On Windows PowerShell:

```powershell
Get-Item C:\path\to\yolo26s_results\weights\best.pt | Select-Object Name, Length
Compress-Archive -Path C:\path\to\yolo26s_results\* -DestinationPath yolo26s_results_complete.zip
```

The `Length` value for `best.pt` must be greater than zero before upload.

## 1. Install the detector integration

```bash
python -m pip install -e ".[yolo]"
```

Review the Ultralytics license that applies to your use before enabling this
adapter.

## 2. Register the checkpoint

After obtaining the real file:

```bash
legopolitics register-yolo-model \
  --weights /path/to/yolo26s_results/weights/best.pt \
  --destination models/lego_figure_best.pt
```

The command:

1. rejects missing and zero-byte checkpoints;
2. loads the checkpoint through Ultralytics;
3. confirms that it produces spatial predictions;
4. records the task and class map;
5. calculates a SHA-256 fingerprint;
6. copies the checkpoint atomically;
7. writes `models/lego_figure_best.pt.json` with model metadata.

You can inspect without copying:

```bash
legopolitics inspect-yolo-model \
  --weights /path/to/yolo26s_results/weights/best.pt
```

## 3. Confirm the model task

The package needs a detector, segmentation model, pose model, or oriented-box
model to locate figures inside complete frames. A pure whole-image
classification checkpoint cannot provide bounding boxes and therefore cannot
extract figure crops.

The presence of files such as `BoxP_curve.png` and `BoxR_curve.png` in a normal
Ultralytics run often accompanies object-detection training, but the actual
checkpoint must be inspected before its task can be confirmed.

## 4. Configure the detector

```yaml
detection:
  enabled: true
  primary_backend: yolo
  yolo:
    weights: models/lego_figure_best.pt
    task: auto
    confidence_threshold: 0.30
    iou_threshold: 0.50
    image_size: 1280
    device: auto
    half_precision: true
    class_filter: null
    save_annotated_frames: true
```

Keep `task: auto` for the first inspection. After the package reports the
checkpoint task, setting it explicitly makes accidental checkpoint replacement
easier to detect.

## 5. Run a short validation video

```bash
legopolitics detect \
  --video examples/test_video.mp4 \
  --config configs/lego_figure_detector.yaml \
  --output outputs/detector_test
```

Review:

- `annotated_frames/`;
- `crops/`;
- the `Detections` worksheet;
- detection confidence distributions;
- false positives and false negatives;
- the validation sample.

## 6. Tune by GPU capability

YOLO and the VLMs have independent device settings. A less powerful GPU can run
YOLO with a smaller `image_size`, lower batch size, and sequential VLM loading.
A more powerful GPU can use larger images and higher-quality VLM profiles.

Suggested starting points:

| GPU class | YOLO image size | Inference batch | Half precision |
|---|---:|---:|---|
| CPU | 640 | 1 | false |
| 4–6 GB GPU | 640 | 1 | true |
| 8–12 GB GPU | 960 | 1–4 | true |
| 16–24 GB GPU | 1280 | 4–8 | true |
| 24+ GB GPU | 1280–1536 | validate empirically | true |

These are starting configurations, not performance guarantees. Benchmark the
real model and video resolution on each machine.

## 7. Do not publish the checkpoint automatically

Keep model files under `models/`, which is excluded from normal source control.
Publish a checkpoint only after confirming that the training images, labels,
base model, and resulting weights may legally be redistributed.

## Supplied YOLO26s LEGO-figure detector

The repository bundle includes a validated user-supplied checkpoint at:

```text
model_assets/lego_figure_yolo26s/best.pt
```

It is a detection model trained at 640 pixels with two classes:

| Class ID | Class name | Default use |
|---:|---|---|
| 0 | `human` | Optional comparison class |
| 1 | `lego_figure` | Default crop and tracking target |

Use the ready configuration:

```bash
legopolitics analyze-video \
  --video videos/example.mp4 \
  --config configs/lego_figure_yolo26s.yaml \
  --output outputs/example
```

The class filter accepts either IDs or class names. The recommended setting is:

```yaml
detection:
  yolo:
    class_filter: [lego_figure]
```

To retain both categories:

```yaml
class_filter: [human, lego_figure]
```

The checkpoint is kept outside `src/`, so it is not bundled in the PyPI wheel. Use Git LFS for public GitHub distribution and confirm the checkpoint's dataset and Ultralytics licensing before redistribution.
