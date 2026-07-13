from __future__ import annotations

from legopolitics.config import AnalysisConfig
from legopolitics.pipeline.stages import PipelineStage


def planned_stages(config: AnalysisConfig) -> list[PipelineStage]:
    stages = [PipelineStage.PROBE, PipelineStage.SAMPLE]
    if config.detection.enabled:
        stages.append(PipelineStage.DETECT)
    if config.segmentation.enabled:
        stages.append(PipelineStage.SEGMENT)
    if config.tracking.enabled:
        stages.append(PipelineStage.TRACK)
    if any(bool(model.get("enabled")) for model in config.vision.models.model_dump().values()):
        stages.append(PipelineStage.VISION)
    if config.ocr.enabled:
        stages.append(PipelineStage.OCR)
    if config.audio.enabled:
        stages.append(PipelineStage.AUDIO)
    stages.extend([PipelineStage.ANALYZE, PipelineStage.VALIDATE, PipelineStage.EXPORT])
    return stages
