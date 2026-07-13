from enum import Enum


class PipelineStage(str, Enum):
    PROBE = "probe"
    SAMPLE = "sample"
    DETECT = "detect"
    SEGMENT = "segment"
    TRACK = "track"
    VISION = "vision"
    OCR = "ocr"
    AUDIO = "audio"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    EXPORT = "export"
