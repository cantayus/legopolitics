from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legopolitics.detection.base import ImageInput
from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError, ModelLoadError
from legopolitics.schemas import BoundingBox, DetectionBatch, DetectionRecord, ModelProvenance
from legopolitics.utils.fingerprints import sha256_file


class YoloDetector:
    def __init__(
        self,
        weights: str | Path,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        image_size: int = 1280,
        device: str = "auto",
        half_precision: bool = True,
        class_filter: list[int | str] | None = None,
        task: str = "auto",
    ) -> None:
        self.weights = Path(weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.half_precision = half_precision
        self.class_filter = class_filter
        self.task = task
        self.model: Any = None
        self.weight_hash = sha256_file(self.weights) if self.weights.exists() else None

    def load(self) -> None:
        if not self.weights.exists():
            raise ModelLoadError(f"YOLO weights do not exist: {self.weights}")
        if not self.weights.is_file():
            raise ModelLoadError(f"YOLO weights path is not a file: {self.weights}")
        if self.weights.stat().st_size == 0:
            raise ModelLoadError(
                f"YOLO weights file is empty: {self.weights}. "
                "Re-copy or re-upload the actual best.pt checkpoint; macOS cloud placeholders "
                "and Finder metadata files are not valid model weights."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DependencyUnavailableError(
                "The Ultralytics adapter requires `pip install legopolitics[yolo]`. "
                "Review Ultralytics licensing before enabling it."
            ) from exc
        try:
            self.model = YOLO(str(self.weights))
        except Exception as exc:
            raise ModelLoadError(f"Could not load YOLO weights {self.weights}: {exc}") from exc
        detected_task = str(getattr(self.model, "task", "unknown"))
        if self.task != "auto" and detected_task != self.task:
            raise ModelLoadError(
                f"Configured YOLO task is {self.task!r}, but the checkpoint reports "
                f"{detected_task!r}."
            )
        if detected_task == "classify":
            raise ModelLoadError(
                "This checkpoint is an image-classification model and does not produce "
                "bounding boxes. legopolitics needs a YOLO detection, segmentation, pose, "
                "or OBB checkpoint to extract figures from whole frames."
            )

    def _resolve_class_filter(self) -> list[int]:
        """Resolve configured class IDs or names against the loaded checkpoint."""
        if self.model is None:
            raise ModelLoadError("YOLO model must be loaded before resolving class filters.")
        names = getattr(self.model, "names", {})
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        normalized = {str(name).strip().lower(): int(index) for index, name in dict(names).items()}
        resolved: list[int] = []
        for value in self.class_filter or []:
            if isinstance(value, int):
                if value not in dict(names):
                    raise ModelLoadError(
                        f"Configured YOLO class ID {value} is not present. Available classes: {names}"
                    )
                resolved.append(value)
                continue
            key = str(value).strip().lower()
            if key not in normalized:
                raise ModelLoadError(
                    f"Configured YOLO class name {value!r} is not present. Available classes: {names}"
                )
            resolved.append(normalized[key])
        return sorted(set(resolved))

    def predict(
        self, images: Sequence[ImageInput], frame_ids: Sequence[str], video_id: str
    ) -> list[DetectionBatch]:
        if self.model is None:
            self.load()
        sources = [str(x) if isinstance(x, Path) else x for x in images]
        kwargs: dict[str, Any] = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.class_filter is not None:
            kwargs["classes"] = self._resolve_class_filter()
        if self.half_precision:
            kwargs["half"] = True
        try:
            results = self.model.predict(source=sources, **kwargs)
        except Exception as exc:
            raise ModelInferenceError(f"YOLO inference failed: {exc}") from exc
        output: list[DetectionBatch] = []
        for frame_id, result in zip(frame_ids, results, strict=True):
            height, width = result.orig_shape
            names = result.names
            detections: list[DetectionRecord] = []
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for index, (coords, confidence, class_id) in enumerate(
                    zip(xyxy, confs, classes, strict=True)
                ):
                    x1, y1, x2, y2 = map(float, coords)
                    box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    keypoints = None
                    if getattr(result, "keypoints", None) is not None:
                        try:
                            keypoints = result.keypoints.xy[index].detach().cpu().numpy().tolist()
                        except (IndexError, AttributeError):
                            keypoints = None
                    detections.append(
                        DetectionRecord(
                            detection_id=f"det_{uuid.uuid4().hex[:16]}",
                            video_id=video_id,
                            frame_id=frame_id,
                            detector_id="ultralytics_yolo",
                            class_id=int(class_id),
                            class_name=str(names[int(class_id)]),
                            confidence=float(confidence),
                            box=box,
                            normalized_box=BoundingBox(
                                x1=x1 / width, y1=y1 / height, x2=x2 / width, y2=y2 / height
                            ),
                            frame_area_fraction=box.area / float(width * height),
                            keypoints=keypoints,
                            model_id=str(self.weights),
                            weight_hash=self.weight_hash,
                        )
                    )
            output.append(DetectionBatch(frame_id=frame_id, detections=detections))
        return output

    def metadata(self) -> ModelProvenance:
        task = getattr(self.model, "task", None) if self.model is not None else None
        return ModelProvenance(
            adapter="ultralytics_yolo",
            model_id=str(self.weights),
            weight_hash=self.weight_hash,
            device=self.device,
            metadata={
                "task": task,
                "classes": getattr(self.model, "names", {}) if self.model is not None else {},
                "class_filter": self.class_filter,
                "license_review_required": True,
            },
        )

    def unload(self) -> None:
        self.model = None


def inspect_yolo_weights(weights: str | Path) -> dict[str, Any]:
    """Load a YOLO checkpoint and return task and class metadata."""
    path = Path(weights)
    detector = YoloDetector(path)
    detector.load()
    assert detector.model is not None
    names = getattr(detector.model, "names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    return {
        "weights": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": detector.weight_hash,
        "task": str(getattr(detector.model, "task", "unknown")),
        "class_count": len(names),
        "classes": {str(key): str(value) for key, value in dict(names).items()},
    }


def register_yolo_weights(
    source: str | Path, destination: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Validate, copy, inspect, and register a custom YOLO checkpoint.

    A JSON sidecar is written next to the copied checkpoint. The checkpoint is
    never embedded in the Python wheel.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists() or not source_path.is_file():
        raise ModelLoadError(f"YOLO source checkpoint does not exist: {source_path}")
    if source_path.stat().st_size == 0:
        raise ModelLoadError(
            f"YOLO source checkpoint is empty: {source_path}. Re-upload the actual .pt file."
        )
    if destination_path.exists() and not overwrite:
        raise ModelLoadError(
            f"Destination already exists: {destination_path}. Use overwrite=True to replace it."
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    shutil.copy2(source_path, temporary)
    temporary.replace(destination_path)
    metadata = inspect_yolo_weights(destination_path)
    sidecar = destination_path.with_suffix(destination_path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {**metadata, "metadata_file": str(sidecar.resolve())}
