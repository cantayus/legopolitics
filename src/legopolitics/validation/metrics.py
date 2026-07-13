from __future__ import annotations


def classification_metrics(
    y_true: list[str], y_pred: list[str], positive_label: str | None = None
) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    accuracy = (
        sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true) if y_true else 0.0
    )
    result = {"accuracy": accuracy}
    if positive_label is not None:
        tp = sum(
            a == positive_label and b == positive_label for a, b in zip(y_true, y_pred, strict=True)
        )
        fp = sum(
            a != positive_label and b == positive_label for a, b in zip(y_true, y_pred, strict=True)
        )
        fn = sum(
            a == positive_label and b != positive_label for a, b in zip(y_true, y_pred, strict=True)
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update(
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        )
    return result


def detection_iou_metrics(ious: list[float], threshold: float = 0.5) -> dict[str, float]:
    return {
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "match_rate": sum(v >= threshold for v in ious) / len(ious) if ious else 0.0,
    }
