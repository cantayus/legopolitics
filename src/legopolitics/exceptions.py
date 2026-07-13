class LegoPoliticsError(Exception):
    """Base exception for all package-specific failures."""


class ConfigurationError(LegoPoliticsError):
    """Raised when configuration is invalid or unsafe."""


class DependencyUnavailableError(LegoPoliticsError):
    """Raised when an optional dependency required by an enabled adapter is absent."""


class ModelLoadError(LegoPoliticsError):
    """Raised when a model or weight file cannot be loaded."""


class ModelInferenceError(LegoPoliticsError):
    """Raised when inference fails after adapter-level handling."""


class VideoProbeError(LegoPoliticsError):
    """Raised when video metadata cannot be obtained."""


class VideoDecodeError(LegoPoliticsError):
    """Raised when a video stream cannot be decoded."""


class CorruptFrameError(LegoPoliticsError):
    """Raised when a decoded frame is corrupt or unusable."""


class AudioExtractionError(LegoPoliticsError):
    """Raised when the source audio stream cannot be extracted."""


class StructuredOutputError(LegoPoliticsError):
    """Raised when a model response cannot be validated as structured output."""


class ManifestError(LegoPoliticsError):
    """Raised for invalid or inconsistent manifest operations."""


class OutputWriteError(LegoPoliticsError):
    """Raised when an artifact cannot be written atomically or validated."""


class LicenseAcknowledgementError(LegoPoliticsError):
    """Raised when a required third-party license acknowledgement is missing."""


class TrademarkReviewError(LegoPoliticsError):
    """Raised when a public release is attempted without documented review."""
