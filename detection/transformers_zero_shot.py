from legopolitics.detection.grounding_dino import GroundingDinoDetector


class TransformersZeroShotDetector(GroundingDinoDetector):
    """Generic Transformers zero-shot detector; model_id controls the backend model."""
