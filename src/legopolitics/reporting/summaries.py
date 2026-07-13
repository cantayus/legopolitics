from typing import Any


def result_summary(result: Any) -> dict[str, Any]:
    return {
        "video_id": result.video.video_id,
        "frames": len(result.frames),
        "detections": len(result.detections),
        "tracks": len(result.tracks),
        "errors": len(result.errors),
    }
