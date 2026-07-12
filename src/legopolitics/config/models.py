from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from legopolitics.constants import CONFIG_SCHEMA_VERSION, SUPPORTED_VIDEO_EXTENSIONS
from legopolitics.exceptions import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(StrictModel):
    name: str = "Political brick-video analysis"
    output_root: Path = Path("outputs")
    overwrite: bool = False
    resume: bool = True
    random_seed: int = 2026


class LegalConfig(StrictModel):
    show_non_affiliation_disclaimer: bool = True
    trademark_review_required_for_public_release: bool = True
    trademark_review_file: Path = Path("TRADEMARK_REVIEW.md")
    ultralytics_license_acknowledged: bool = False
    allow_development_release_override: bool = False
    public_distribution_name: str = "legopolitics"


class InputConfig(StrictModel):
    video: Path | None = None
    recursive: bool = False
    allowed_extensions: list[str] = Field(
        default_factory=lambda: sorted(SUPPORTED_VIDEO_EXTENSIONS)
    )

    @field_validator("allowed_extensions")
    @classmethod
    def normalize_extensions(cls, values: list[str]) -> list[str]:
        return sorted({v.lower() if v.startswith(".") else f".{v.lower()}" for v in values})


class SamplingConfig(StrictModel):
    mode: Literal["fixed", "scene", "hybrid", "adaptive", "all"] = "hybrid"
    fps: float = 1.0
    include_first_frame: bool = True
    include_last_frame: bool = True
    include_scene_boundaries: bool = True
    frames_per_scene: int = 1
    scene_detector: Literal["adaptive", "content", "simple"] = "adaptive"
    scene_threshold: float | None = None
    motion_sampling: bool = True
    motion_threshold: float = 0.30
    maximum_gap_seconds: float = 2.0
    minimum_gap_seconds: float = 0.10
    deduplicate_frames: bool = True
    perceptual_hash_threshold: int = 4
    all_mode_safety_max_frames: int = 1_000_000

    @model_validator(mode="after")
    def validate_sampling(self) -> SamplingConfig:
        if self.fps <= 0:
            raise ValueError("sampling.fps must be greater than zero")
        if self.minimum_gap_seconds < 0:
            raise ValueError("minimum_gap_seconds cannot be negative")
        if self.maximum_gap_seconds < self.minimum_gap_seconds:
            raise ValueError("maximum_gap_seconds must be >= minimum_gap_seconds")
        return self


class YoloConfig(StrictModel):
    weights: Path | None = None
    task: str = "auto"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.50
    image_size: int = 1280
    device: str = "auto"
    half_precision: bool = True
    class_filter: list[int | str] | None = None
    save_annotated_frames: bool = True


class OpenVocabularyConfig(StrictModel):
    enabled: bool = False
    backend: str = "grounding_dino"
    model_id: str | None = None
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    text_queries: list[str] = Field(default_factory=list)


class DetectionConfig(StrictModel):
    enabled: bool = False
    primary_backend: str = "mock"
    yolo: YoloConfig = Field(default_factory=YoloConfig)
    open_vocabulary: OpenVocabularyConfig = Field(default_factory=OpenVocabularyConfig)
    fusion_iou_threshold: float = 0.55


class SegmentationConfig(StrictModel):
    enabled: bool = False
    backend: str = "sam2"
    model_id: str | None = None
    use_detection_boxes_as_prompts: bool = True
    propagate_masks: bool = True
    save_masks: bool = True
    save_rgba_crops: bool = True


class TrackingConfig(StrictModel):
    enabled: bool = True
    backend: str = "iou"
    tracker: str = "bytetrack"
    use_appearance_embeddings: bool = False
    appearance_model: str = "dinov2"
    secondary_appearance_model: str = "clip"
    maximum_track_gap_frames: int = 10
    minimum_track_length_frames: int = 1
    iou_match_threshold: float = 0.30
    enable_track_merging: bool = True
    enable_track_splitting_checks: bool = True


class ModelConfig(StrictModel):
    name: str | None = None
    enabled: bool = False
    model_id: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    tasks: list[str] = Field(default_factory=list)
    backend: str | None = None
    labels_from_codebook: bool = False
    device: str = "auto"
    dtype: str = "auto"
    generation: dict[str, Any] = Field(default_factory=dict)
    input_mode: Literal["auto", "plain", "chat"] = "auto"
    processor_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    pipeline_kwargs: dict[str, Any] = Field(default_factory=dict)
    quantization: Literal["none", "4bit", "8bit"] = "none"
    device_map: str | dict[str, int | str] | None = None
    max_memory: dict[str, int | str] = Field(default_factory=dict)
    offload_folder: str | None = None
    use_flash_attention_2: bool = False


class VisionModelsConfig(StrictModel):
    florence2: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            enabled=False,
            model_id="microsoft/Florence-2-large-ft",
            tasks=["more_detailed_caption", "ocr_with_region"],
        )
    )
    blip: ModelConfig = Field(default_factory=ModelConfig)
    blip2: ModelConfig = Field(default_factory=ModelConfig)
    clip: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=False, model_id="openai/clip-vit-large-patch14")
    )
    dinov2: ModelConfig = Field(
        default_factory=lambda: ModelConfig(enabled=False, model_id="facebook/dinov2-large")
    )
    general_vlm: ModelConfig = Field(default_factory=ModelConfig)
    huggingface_vlms: list[ModelConfig] = Field(default_factory=list)
    video_vlm: ModelConfig = Field(default_factory=ModelConfig)


class VisionConfig(StrictModel):
    whole_frame: bool = True
    crops: bool = True
    context_crops: bool = True
    track_representatives: bool = True
    track_contact_sheets: bool = True
    sequence_windows: bool = True
    sequence_window_frames: int = 5
    models: VisionModelsConfig = Field(default_factory=VisionModelsConfig)


class OCRConfig(StrictModel):
    enabled: bool = False
    backends: list[str] = Field(default_factory=lambda: ["florence2"])
    detect_language: bool = True
    translate_to: str | None = "en"
    temporal_deduplication: bool = True
    associate_text_with_tracks: bool = True
    similarity_threshold: float = 0.90


class ASRConfig(StrictModel):
    backend: str = "whisper"
    model: str = "openai/whisper-large-v3"
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    word_timestamps: bool = True


class DiarizationConfig(StrictModel):
    enabled: bool = False
    backend: str = "pyannote"
    model_id: str | None = "pyannote/speaker-diarization-community-1"


class TranslationConfig(StrictModel):
    enabled: bool = False
    target_language: str = "en"
    model_id: str | None = None


class AudioEventsConfig(StrictModel):
    enabled: bool = False
    model_id: str | None = None
    labels: list[str] = Field(default_factory=list)


class AudioConfig(StrictModel):
    enabled: bool = False
    extract_audio: bool = True
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    audio_events: AudioEventsConfig = Field(default_factory=AudioEventsConfig)


class KnownGroup(StrictModel):
    group_id: str
    label: str
    descriptions: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    reference_images: list[Path] = Field(default_factory=list)


class ResearchConfig(StrictModel):
    event_context: str = ""
    research_question: str = ""
    known_groups: list[KnownGroup] = Field(default_factory=list)
    codebook_path: Path | None = None
    inference_constraints: list[str] = Field(
        default_factory=lambda: [
            "Do not infer nationality solely from clothing color.",
            "Do not infer religion from appearance.",
            "Do not infer ethnicity from depicted skin or head color.",
            "Separate observation from interpretation.",
            "Do not treat supplied context as visual evidence.",
        ]
    )


class PromptingConfig(StrictModel):
    blind_visual_pass: bool = True
    contextual_pass: bool = True
    codebook_pass: bool = True
    contradiction_pass: bool = True
    relationship_pass: bool = True
    narrative_pass: bool = True
    save_rendered_prompts: bool = True
    save_raw_responses: bool = True
    maximum_retries: int = 2
    structured_output: bool = True


class OutputConfig(StrictModel):
    excel: bool = True
    parquet: bool = True
    jsonl: bool = True
    sqlite: bool = True
    html_report: bool = True
    save_frames: bool = True
    save_crops: bool = True
    save_masks: bool = True
    save_annotated_frames: bool = True
    one_workbook_per_video: bool = True
    global_dataset_tables: bool = False
    include_excel_charts: bool = True
    strict_parquet: bool = False


class PerformanceConfig(StrictModel):
    device: str = "auto"
    dtype: str = "auto"
    hardware_profile: Literal[
        "auto", "cpu", "small_gpu", "midrange_gpu", "high_end_gpu", "maximum"
    ] = "auto"
    enable_tf32: bool = True
    inference_batch_size: int | Literal["auto"] = "auto"
    cpu_workers: int = 4
    prefetch_frames: int = 16
    pin_memory: bool = True
    low_vram_mode: bool = False
    unload_models_between_stages: bool = False
    oom_recovery: bool = True
    allow_cpu_fallback: bool = True


class LoggingConfig(StrictModel):
    level: str = "INFO"
    console: bool = True
    file: bool = True
    json_log: bool = True
    progress_bar: bool = True
    scan_progress_every: int = 1000
    process_progress_every: int = 100


class ValidationConfig(StrictModel):
    enabled: bool = True
    random_sample_size: int = 100
    low_confidence_sample_size: int = 100
    disagreement_sample_size: int = 100
    rare_class_sample_size: int = 50
    stratify_by_video: bool = True
    stratify_by_class: bool = True


class AnalysisConfig(StrictModel):
    schema_version: str = CONFIG_SCHEMA_VERSION
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    legal: LegalConfig = Field(default_factory=LegalConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    prompting: PromptingConfig = Field(default_factory=PromptingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisConfig:
        source = Path(path)
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            if not isinstance(payload, dict):
                raise ConfigurationError("Configuration root must be a mapping")
            return cls.model_validate(payload)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"Could not load configuration {source}: {exc}") from exc

    @classmethod
    def from_json(cls, path: str | Path) -> AnalysisConfig:
        source = Path(path)
        try:
            return cls.model_validate_json(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"Could not load configuration {source}: {exc}") from exc

    def to_yaml(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(self.model_dump_json())
        target.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return target

    def config_hash(self) -> str:
        canonical = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_overrides(self, **overrides: Any) -> AnalysisConfig:
        data = self.model_dump(mode="python")
        for dotted_key, value in overrides.items():
            cursor = data
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                cursor = cursor[part]
            cursor[parts[-1]] = value
        return AnalysisConfig.model_validate(data)
