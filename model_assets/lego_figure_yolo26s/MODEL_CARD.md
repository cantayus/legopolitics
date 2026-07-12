# LEGO Figure YOLO26s Detector

This user-supplied checkpoint is integrated as the default spatial detector for the research repository. It is not included in the PyPI wheel.

## Model details

- Task: object detection
- Architecture: YOLO26s
- Input size used for training: 640 px
- Classes: `human` (0), `lego_figure` (1)
- Default analysis class: `lego_figure` (1)
- Epochs: 50
- Seed: 42
- Checkpoint SHA-256: `b027885bd139f7eced7e39beb4e734ad7fa9a88ef28c77cecea75d17bdc5e98b`

## Final validation metrics

- Precision: 0.94639
- Recall: 0.70621
- mAP50: 0.79407
- mAP50–95: 0.59977

Metrics were read from the supplied training run's final CSV row. They should be independently evaluated on held-out videos from the target research corpus.

## Usage

Use `class_filter: [lego_figure]` to extract only brick figures. Use `class_filter: [human, lego_figure]` when both classes are analytically relevant.

## Distribution and license

The repository source code is licensed under the MIT License, but this checkpoint is **not automatically covered by that license**. It is a user-provided research asset and is excluded from the PyPI wheel. Confirm training-data rights, checkpoint ownership, and applicable Ultralytics licensing before public redistribution. When redistribution is authorized, include this model card and `model_metadata.json` with the checkpoint.
