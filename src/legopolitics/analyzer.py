from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from legopolitics.analysis import (
    aggregate_model_responses,
    frame_composition,
    infer_proximity_relations,
    scenes_to_narrative,
    track_lifecycle_events,
)
from legopolitics.analysis.attributes import crop_attributes
from legopolitics.analysis.salience import compute_salience
from legopolitics.audio.alignment import align_speakers, frames_for_interval
from legopolitics.audio.diarization import PyannoteDiarizationAdapter
from legopolitics.audio.events import AudioEventAdapter
from legopolitics.audio.faster_whisper import FasterWhisperAdapter
from legopolitics.audio.whisper import WhisperAdapter
from legopolitics.config import AnalysisConfig, validate_runtime_config
from legopolitics.constants import PUBLIC_RESULT_SCHEMA_VERSION
from legopolitics.detection import (
    GroundingDinoDetector,
    MockDetector,
    YoloDetector,
    fuse_detections,
    make_crops,
)
from legopolitics.detection.base import DetectorAdapter
from legopolitics.ocr import FlorenceOCRAdapter, TrOCRAdapter, deduplicate_temporally
from legopolitics.ocr.base import OCRAdapter
from legopolitics.ocr.regions import associate_ocr_with_visible_tracks
from legopolitics.pipeline.cache import load_stage_cache, save_stage_cache
from legopolitics.pipeline.context import PipelineContext
from legopolitics.pipeline.orchestrator import run_stage
from legopolitics.pipeline.resume import load_completed_result
from legopolitics.prompting import (
    BLIND_OBSERVATION_TEMPLATE,
    CONTEXTUAL_TEMPLATE,
    Codebook,
)
from legopolitics.prompting.rendering import render
from legopolitics.prompting.versioning import PROMPT_VERSION, prompt_hash
from legopolitics.reporting.html import write_html_report
from legopolitics.schemas import (
    AnalysisResult,
    AttributeRecord,
    AudioEvent,
    CodebookScore,
    DetectionRecord,
    DiarizationSegment,
    EventRecord,
    FrameRecord,
    MaskRecord,
    ModelAgreementRecord,
    NarrativeSegment,
    OCRRecord,
    RelationRecord,
    RunProvenance,
    SceneRecord,
    TrackObservation,
    TrackRecord,
    TranscriptSegment,
    VideoRecord,
    VisualResponse,
)
from legopolitics.segmentation import BoxMaskSegmenter, Sam2Segmenter
from legopolitics.segmentation.base import SegmenterAdapter
from legopolitics.storage import OutputLayout, RunManifest, export_result
from legopolitics.storage.atomic import atomic_write_text
from legopolitics.storage.jsonl import append_jsonl
from legopolitics.tracking import IoUTracker
from legopolitics.utils.device import detect_device
from legopolitics.utils.environment import inspect_environment
from legopolitics.utils.fingerprints import stable_hash
from legopolitics.utils.logging import configure_logging
from legopolitics.validation import sample_for_validation
from legopolitics.version import __version__
from legopolitics.video import discover_videos, probe_video, sample_video
from legopolitics.video.audio_extract import extract_audio
from legopolitics.vision import (
    Blip2Adapter,
    BlipAdapter,
    ClipAdapter,
    DinoV2Adapter,
    Florence2Adapter,
    HuggingFaceVLMAdapter,
    VisionRequest,
)
from legopolitics.vision.base import VisionModelAdapter

AnalysisArtifacts = tuple[
    list[AttributeRecord],
    list[RelationRecord],
    list[EventRecord],
    list[NarrativeSegment],
    list[CodebookScore],
    list[ModelAgreementRecord],
]


class LegoPoliticsAnalyzer:
    def __init__(self, config: AnalysisConfig | None = None, **kwargs: Any) -> None:
        self.config = config or AnalysisConfig()
        if kwargs:
            self._apply_direct_kwargs(kwargs)

    def _apply_direct_kwargs(self, kwargs: dict[str, Any]) -> None:
        mapping = {
            "yolo_weights": "detection.yolo.weights",
            "sampling_mode": "sampling.mode",
            "sample_fps": "sampling.fps",
            "enable_scene_detection": "sampling.include_scene_boundaries",
            "enable_open_vocabulary_detection": "detection.open_vocabulary.enabled",
            "enable_tracking": "tracking.enabled",
            "enable_segmentation": "segmentation.enabled",
            "enable_audio": "audio.enabled",
            "enable_ocr": "ocr.enabled",
            "enable_blind_pass": "prompting.blind_visual_pass",
            "enable_contextual_pass": "prompting.contextual_pass",
            "enable_codebook_pass": "prompting.codebook_pass",
            "enable_contradiction_pass": "prompting.contradiction_pass",
        }
        overrides = {mapping[key]: value for key, value in kwargs.items() if key in mapping}
        if "yolo_weights" in kwargs:
            overrides["detection.enabled"] = True
            overrides["detection.primary_backend"] = "yolo"
        unknown = set(kwargs) - set(mapping)
        if unknown:
            raise TypeError(f"Unknown analyzer arguments: {sorted(unknown)}")
        self.config = self.config.with_overrides(**overrides)

    def analyze_video(self, video_path: Path, output_dir: Path) -> AnalysisResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        validate_runtime_config(self.config, video_path)
        layout = OutputLayout.create(output_dir)
        logger = configure_logging(
            self.config.logging.level,
            layout.logs,
            self.config.logging.console,
            self.config.logging.json_log,
        )
        video = probe_video(video_path)
        cache_key = stable_hash(
            {
                "video": video.fingerprint,
                "config": self.config.config_hash(),
                "schema": PUBLIC_RESULT_SCHEMA_VERSION,
            }
        )
        completed = (
            load_completed_result(
                layout.root / "result.json", self.config.config_hash(), video.fingerprint
            )
            if self.config.project.resume
            else None
        )
        if completed:
            logger.info(
                "Returning completed result with matching video and configuration fingerprints"
            )
            return completed
        with RunManifest(layout.root / "run_manifest.sqlite") as manifest:
            run_id = manifest.begin_run(
                self.config.config_hash(),
                {"video": str(video_path), "package_version": __version__},
            )
            manifest.record_config(
                self.config.config_hash(),
                self.config.schema_version,
                self.config.model_dump(mode="json"),
            )
            manifest.record_video(
                video.video_id,
                str(video.path),
                video.fingerprint,
                metadata=video.model_dump(mode="json"),
            )
            context = PipelineContext(self.config, layout, manifest, run_id, logger)
            run_config_path = self.config.to_yaml(layout.root / "run_config.yaml")
            effective_config_path = self.config.to_yaml(layout.root / "effective_config.yaml")
            try:
                scenes, frames = self._sample(context, video, cache_key)
                detections, fused = self._detect(context, video, frames, cache_key)
                masks = self._segment(context, video, frames, fused or detections, cache_key)
                tracks, observations = self._track(
                    context, video, frames, fused or detections, cache_key
                )
                visual = self._vision(
                    context, video, frames, fused or detections, tracks, cache_key
                )
                ocr = self._ocr(context, video, frames, observations, cache_key)
                transcript, diarization, audio_events = self._audio(
                    context, video, frames, cache_key
                )
                attributes, relations, events, narratives, codebook_scores, agreements = (
                    self._analyze(
                        context,
                        video,
                        frames,
                        scenes,
                        fused or detections,
                        tracks,
                        observations,
                        visual,
                        transcript,
                        ocr,
                        cache_key,
                    )
                )
                provenance = self._provenance(run_id)
                result = AnalysisResult(
                    video=video,
                    scenes=scenes,
                    frames=frames,
                    detections=detections,
                    fused_detections=fused,
                    masks=masks,
                    tracks=tracks,
                    track_observations=observations,
                    attributes=attributes,
                    visual_responses=visual,
                    ocr=ocr,
                    transcript=transcript,
                    diarization=diarization,
                    audio_events=audio_events,
                    relations=relations,
                    events=events,
                    narrative_segments=narratives,
                    codebook_scores=codebook_scores,
                    model_agreement=agreements,
                    errors=context.errors,
                    provenance=provenance,
                    output_paths=layout.output_paths(),
                )
                result.validation_summary = (
                    sample_for_validation(
                        result,
                        self.config.validation.random_sample_size,
                        self.config.validation.low_confidence_sample_size,
                        self.config.validation.disagreement_sample_size,
                        self.config.project.random_seed,
                    )
                    if self.config.validation.enabled
                    else []
                )
                started = manifest.begin_stage(run_id, video.video_id, "export", cache_key)
                artifacts = export_result(result, self.config)
                artifacts["run_config"] = run_config_path
                artifacts["effective_config"] = effective_config_path
                artifacts["environment"] = atomic_write_text(
                    layout.root / "environment.json",
                    json.dumps(inspect_environment(), ensure_ascii=False, indent=2, default=str),
                )
                artifacts["provenance"] = atomic_write_text(
                    layout.root / "provenance.json", provenance.model_dump_json(indent=2)
                )
                if self.config.output.html_report:
                    artifacts["report"] = write_html_report(layout.root / "report.html", result)
                atomic_write_text(layout.root / "result.json", result.model_dump_json(indent=2))
                for kind, path in artifacts.items():
                    manifest.register_artifact(run_id, video.video_id, "export", path, kind)
                manifest.finish_stage(
                    run_id,
                    video.video_id,
                    "export",
                    started,
                    metadata={"artifacts": len(artifacts)},
                )
                manifest.finish_run(run_id, "complete")
                return result
            except Exception:
                manifest.finish_run(run_id, "failed")
                raise

    def analyze_directory(
        self,
        input_root: Path,
        output_root: Path,
        recursive: bool = True,
        preserve_directory_structure: bool = True,
        num_batches: int = 1,
        batch_id: int = 0,
        resume: bool = True,
    ) -> list[AnalysisResult]:
        config = self.config.with_overrides(**{"project.resume": resume})
        analyzer = LegoPoliticsAnalyzer(config)
        source = Path(input_root)
        results = []
        videos = discover_videos(
            source,
            recursive,
            set(config.input.allowed_extensions),
            config.logging.scan_progress_every,
            num_batches,
            batch_id,
        )
        for video in videos:
            relative = (
                video.relative_to(source)
                if source.is_dir() and preserve_directory_structure
                else Path(video.name)
            )
            target = Path(output_root) / relative.parent / video.stem
            try:
                results.append(analyzer.analyze_video(video, target))
            except Exception as exc:
                configure_logging(config.logging.level).exception("Video failed %s: %s", video, exc)
        return results

    def _sample(
        self, context: PipelineContext, video: VideoRecord, key: str
    ) -> tuple[list[SceneRecord], list[FrameRecord]]:
        path = context.layout.cache / "sample.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [SceneRecord.model_validate(x) for x in cached["scenes"]], [
                FrameRecord.model_validate(x) for x in cached["frames"]
            ]
        value = run_stage(
            context,
            video.video_id,
            "sample",
            key,
            lambda: sample_video(video, context.layout.root, context.config),
        )
        assert value is not None
        scenes, frames = value
        save_stage_cache(
            path,
            key,
            {
                "scenes": [x.model_dump(mode="json") for x in scenes],
                "frames": [x.model_dump(mode="json") for x in frames],
            },
        )
        return scenes, frames

    def _detector(self) -> DetectorAdapter:
        cfg = self.config.detection
        if cfg.primary_backend == "yolo":
            assert cfg.yolo.weights is not None
            return YoloDetector(
                cfg.yolo.weights,
                cfg.yolo.confidence_threshold,
                cfg.yolo.iou_threshold,
                cfg.yolo.image_size,
                cfg.yolo.device,
                cfg.yolo.half_precision,
                cfg.yolo.class_filter,
                cfg.yolo.task,
            )
        return MockDetector()

    def _detect(
        self, context: PipelineContext, video: VideoRecord, frames: list[FrameRecord], key: str
    ) -> tuple[list[DetectionRecord], list[DetectionRecord]]:
        if not context.config.detection.enabled:
            return [], []
        path = context.layout.cache / "detection.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [DetectionRecord.model_validate(x) for x in cached["detections"]], [
                DetectionRecord.model_validate(x) for x in cached["fused"]
            ]

        def compute() -> tuple[list[DetectionRecord], list[DetectionRecord]]:
            images = [f.image_path for f in frames if f.image_path]
            ids = [f.frame_id for f in frames if f.image_path]
            detector = self._detector()
            batches = detector.predict(images, ids, video.video_id)
            detections = [d for batch in batches for d in batch.detections]
            if context.config.detection.open_vocabulary.enabled:
                ocfg = context.config.detection.open_vocabulary
                adapter = GroundingDinoDetector(
                    ocfg.model_id or "IDEA-Research/grounding-dino-tiny",
                    ocfg.text_queries,
                    ocfg.box_threshold,
                    context.config.performance.device,
                )
                open_batches = adapter.predict(images, ids, video.video_id)
                detections.extend(d for batch in open_batches for d in batch.detections)
            by_frame = {f.frame_id: f for f in frames}
            for frame_id in ids:
                frame_dets = [d for d in detections if d.frame_id == frame_id]
                frame = by_frame[frame_id]
                if frame.image_path:
                    make_crops(frame.image_path, frame_dets, context.layout.crops)
            fused = fuse_detections(detections, context.config.detection.fusion_iou_threshold)
            for item in fused:
                source = next(
                    (
                        d
                        for d in detections
                        if d.detection_id in item.source_detection_ids and d.crop_path
                    ),
                    None,
                )
                if source:
                    item.crop_path = source.crop_path
                    item.context_crop_path = source.context_crop_path
            return detections, fused

        value = run_stage(context, video.video_id, "detect", key, compute, recoverable=True)
        if value is None:
            return [], []
        detections, fused = value
        save_stage_cache(
            path,
            key,
            {
                "detections": [x.model_dump(mode="json") for x in detections],
                "fused": [x.model_dump(mode="json") for x in fused],
            },
        )
        return detections, fused

    def _segment(
        self,
        context: PipelineContext,
        video: VideoRecord,
        frames: list[FrameRecord],
        detections: list[DetectionRecord],
        key: str,
    ) -> list[MaskRecord]:
        if not context.config.segmentation.enabled or not detections:
            return []
        path = context.layout.cache / "masks.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [MaskRecord.model_validate(x) for x in cached]

        def compute() -> list[MaskRecord]:
            adapter: SegmenterAdapter = (
                Sam2Segmenter(
                    context.config.segmentation.model_id or "facebook/sam2-hiera-small",
                    context.config.performance.device,
                )
                if context.config.segmentation.backend == "sam2"
                else BoxMaskSegmenter()
            )
            output = []
            for frame in frames:
                if frame.image_path:
                    output.extend(
                        adapter.segment(
                            frame.image_path,
                            [d for d in detections if d.frame_id == frame.frame_id],
                            context.layout.masks,
                        )
                    )
            return output

        masks = run_stage(context, video.video_id, "segment", key, compute, recoverable=True) or []
        save_stage_cache(path, key, [x.model_dump(mode="json") for x in masks])
        return masks

    def _track(
        self,
        context: PipelineContext,
        video: VideoRecord,
        frames: list[FrameRecord],
        detections: list[DetectionRecord],
        key: str,
    ) -> tuple[list[TrackRecord], list[TrackObservation]]:
        if not context.config.tracking.enabled or not detections:
            return [], []
        path = context.layout.cache / "tracking.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [TrackRecord.model_validate(x) for x in cached["tracks"]], [
                TrackObservation.model_validate(x) for x in cached["observations"]
            ]
        tracker = IoUTracker(
            context.config.tracking.iou_match_threshold,
            context.config.tracking.maximum_track_gap_frames,
        )
        value = run_stage(
            context,
            video.video_id,
            "track",
            key,
            lambda: tracker.track(frames, detections),
            recoverable=True,
        )
        if value is None:
            return [], []
        tracks, observations = value
        save_stage_cache(
            path,
            key,
            {
                "tracks": [x.model_dump(mode="json") for x in tracks],
                "observations": [x.model_dump(mode="json") for x in observations],
            },
        )
        return tracks, observations

    def _vision_adapters(self) -> list[VisionModelAdapter]:
        models = self.config.vision.models
        adapters: list[VisionModelAdapter] = []
        if models.florence2.enabled:
            adapters.append(
                Florence2Adapter(
                    models.florence2.model_id or "microsoft/Florence-2-large-ft",
                    models.florence2.revision,
                    self.config.performance.device,
                    self.config.performance.dtype,
                    models.florence2.trust_remote_code,
                )
            )
        if models.blip.enabled:
            adapters.append(
                BlipAdapter(
                    models.blip.model_id or "Salesforce/blip-image-captioning-large",
                    self.config.performance.device,
                )
            )
        if models.blip2.enabled:
            adapters.append(
                Blip2Adapter(
                    models.blip2.model_id or "Salesforce/blip2-opt-2.7b",
                    self.config.performance.device,
                    self.config.performance.dtype,
                )
            )
        if models.clip.enabled:
            adapters.append(
                ClipAdapter(
                    models.clip.model_id or "openai/clip-vit-large-patch14",
                    self.config.performance.device,
                )
            )
        if models.dinov2.enabled:
            adapters.append(
                DinoV2Adapter(
                    models.dinov2.model_id or "facebook/dinov2-large",
                    self.config.performance.device,
                )
            )
        hf_models = []
        if models.general_vlm.enabled and models.general_vlm.model_id:
            hf_models.append(models.general_vlm)
        hf_models.extend(
            model for model in models.huggingface_vlms if model.enabled and model.model_id
        )
        for index, model in enumerate(hf_models, start=1):
            assert model.model_id is not None
            adapters.append(
                HuggingFaceVLMAdapter(
                    model.model_id,
                    model.device if model.device != "auto" else self.config.performance.device,
                    model.trust_remote_code,
                    revision=model.revision,
                    dtype=model.dtype if model.dtype != "auto" else self.config.performance.dtype,
                    input_mode=model.input_mode,
                    generation=model.generation,
                    processor_kwargs=model.processor_kwargs,
                    model_kwargs=model.model_kwargs,
                    pipeline_kwargs=model.pipeline_kwargs,
                    adapter_name=model.name or f"hf_vlm_{index}",
                    hardware_profile=self.config.performance.hardware_profile,
                    quantization=model.quantization,
                    device_map=model.device_map,
                    max_memory=model.max_memory,
                    offload_folder=model.offload_folder,
                    use_flash_attention_2=model.use_flash_attention_2,
                )
            )
        return adapters

    def _vision(
        self,
        context: PipelineContext,
        video: VideoRecord,
        frames: list[FrameRecord],
        detections: list[DetectionRecord],
        tracks: list[TrackRecord],
        key: str,
    ) -> list[VisualResponse]:
        adapters = self._vision_adapters()
        if not adapters:
            return []
        path = context.layout.cache / "vision.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [VisualResponse.model_validate(x) for x in cached]

        def compute() -> list[VisualResponse]:
            output: list[VisualResponse] = []
            prompt_file = context.layout.jsonl / "rendered_prompts.jsonl"
            raw_file = context.layout.jsonl / "raw_model_outputs.jsonl"
            for adapter in adapters:
                provenance = adapter.metadata()
                for frame in frames:
                    if not frame.image_path:
                        continue
                    requests = []
                    if context.config.prompting.blind_visual_pass:
                        requests.append(
                            (
                                "blind",
                                VisionRequest(
                                    prompt=BLIND_OBSERVATION_TEMPLATE,
                                    task="more_detailed_caption"
                                    if provenance.adapter == "florence2"
                                    else "caption",
                                    candidate_labels=[
                                        "figure",
                                        "weapon",
                                        "flag",
                                        "vehicle",
                                        "building",
                                        "text",
                                    ]
                                    if provenance.adapter == "clip"
                                    else [],
                                ),
                            )
                        )
                    if context.config.prompting.contextual_pass:
                        prompt = render(
                            CONTEXTUAL_TEMPLATE,
                            event_context=context.config.research.event_context,
                            research_question=context.config.research.research_question,
                            known_groups=[
                                g.model_dump(mode="json")
                                for g in context.config.research.known_groups
                            ],
                            constraints=context.config.research.inference_constraints,
                        )
                        requests.append(
                            (
                                "contextual",
                                VisionRequest(
                                    prompt=prompt,
                                    task="caption",
                                    candidate_labels=[
                                        "ingroup heroism",
                                        "outgroup threat",
                                        "victimization",
                                        "self-defense",
                                        "victory",
                                        "uncertain",
                                    ]
                                    if provenance.adapter == "clip"
                                    else [],
                                ),
                            )
                        )
                    for pass_type, request in requests:
                        append_jsonl(
                            prompt_file,
                            {
                                "frame_id": frame.frame_id,
                                "pass_type": pass_type,
                                "model": provenance.model_id,
                                "prompt": request.prompt,
                                "prompt_hash": prompt_hash(request.prompt),
                            },
                        )
                        try:
                            response = adapter.analyze_image(frame.image_path, request)
                        except Exception as exc:
                            context.logger.warning(
                                "Vision model %s failed on %s: %s",
                                provenance.model_id,
                                frame.frame_id,
                                exc,
                            )
                            continue
                        append_jsonl(
                            raw_file,
                            {
                                "frame_id": frame.frame_id,
                                "pass_type": pass_type,
                                "model": provenance.model_id,
                                "raw_text": response.raw_text,
                                "parsed": response.parsed,
                            },
                        )
                        output.append(
                            VisualResponse(
                                response_id=f"visual_{uuid.uuid4().hex[:16]}",
                                target_type="frame",
                                target_id=frame.frame_id,
                                pass_type=pass_type,
                                prompt_version=PROMPT_VERSION,
                                raw_text=response.raw_text,
                                parsed=response.parsed,
                                confidence=response.confidence,
                                provenance=adapter.metadata(),
                            )
                        )
                if context.config.vision.crops:
                    for detection in detections:
                        if not detection.crop_path:
                            continue
                        request = VisionRequest(
                            prompt=BLIND_OBSERVATION_TEMPLATE,
                            task="more_detailed_caption"
                            if provenance.adapter == "florence2"
                            else "caption",
                            candidate_labels=[
                                "soldier-like figure",
                                "civilian-like figure",
                                "leader-like figure",
                                "weapon",
                                "flag",
                                "other",
                            ]
                            if provenance.adapter == "clip"
                            else [],
                        )
                        try:
                            response = adapter.analyze_image(detection.crop_path, request)
                        except Exception:
                            continue
                        output.append(
                            VisualResponse(
                                response_id=f"visual_{uuid.uuid4().hex[:16]}",
                                target_type="detection",
                                target_id=detection.detection_id,
                                pass_type="blind",
                                prompt_version=PROMPT_VERSION,
                                raw_text=response.raw_text,
                                parsed=response.parsed,
                                confidence=response.confidence,
                                provenance=adapter.metadata(),
                            )
                        )
                if context.config.performance.unload_models_between_stages:
                    adapter.unload()
            return output

        responses = (
            run_stage(context, video.video_id, "vision", key, compute, recoverable=True) or []
        )
        save_stage_cache(path, key, [x.model_dump(mode="json") for x in responses])
        return responses

    def _ocr(
        self,
        context: PipelineContext,
        video: VideoRecord,
        frames: list[FrameRecord],
        observations: list[TrackObservation],
        key: str,
    ) -> list[OCRRecord]:
        if not context.config.ocr.enabled:
            return []
        path = context.layout.cache / "ocr.json"
        cached = load_stage_cache(path, key)
        if cached:
            return [OCRRecord.model_validate(x) for x in cached]

        def compute() -> list[OCRRecord]:
            adapters: list[OCRAdapter] = []
            for backend in context.config.ocr.backends:
                if backend == "florence2":
                    adapters.append(FlorenceOCRAdapter(device=context.config.performance.device))
                elif backend == "trocr":
                    adapters.append(TrOCRAdapter(device=context.config.performance.device))
            records: list[OCRRecord] = []
            for frame in frames:
                if not frame.image_path:
                    continue
                for adapter in adapters:
                    try:
                        records.extend(
                            adapter.recognize(frame.image_path, frame.frame_id, video.video_id)
                        )
                    except Exception as exc:
                        context.logger.warning("OCR failed for %s: %s", frame.frame_id, exc)
            if context.config.ocr.temporal_deduplication:
                records = deduplicate_temporally(records, context.config.ocr.similarity_threshold)
            if context.config.ocr.associate_text_with_tracks:
                associate_ocr_with_visible_tracks(records, observations)
            return records

        records = run_stage(context, video.video_id, "ocr", key, compute, recoverable=True) or []
        save_stage_cache(path, key, [x.model_dump(mode="json") for x in records])
        return records

    def _audio(
        self, context: PipelineContext, video: VideoRecord, frames: list[FrameRecord], key: str
    ) -> tuple[list[TranscriptSegment], list[DiarizationSegment], list[AudioEvent]]:
        if not context.config.audio.enabled or not video.has_audio:
            return [], [], []
        path = context.layout.cache / "audio.json"
        cached = load_stage_cache(path, key)
        if cached:
            return (
                [TranscriptSegment.model_validate(x) for x in cached["transcript"]],
                [DiarizationSegment.model_validate(x) for x in cached["diarization"]],
                [AudioEvent.model_validate(x) for x in cached["events"]],
            )

        def compute() -> tuple[list[TranscriptSegment], list[DiarizationSegment], list[AudioEvent]]:
            audio_path = (
                extract_audio(video.path, context.layout.audio / "audio.wav")
                if context.config.audio.extract_audio
                else context.layout.audio / "audio.wav"
            )
            asr_cfg = context.config.audio.asr
            asr = (
                FasterWhisperAdapter(
                    asr_cfg.model,
                    context.config.performance.device,
                    language=asr_cfg.language,
                    task=asr_cfg.task,
                )
                if asr_cfg.backend == "faster_whisper"
                else WhisperAdapter(
                    asr_cfg.model,
                    context.config.performance.device,
                    asr_cfg.language,
                    asr_cfg.task,
                    asr_cfg.word_timestamps,
                )
            )
            transcript = asr.transcribe(audio_path, video.video_id)
            diar: list[DiarizationSegment] = []
            events: list[AudioEvent] = []
            if context.config.audio.diarization.enabled:
                diar = PyannoteDiarizationAdapter(
                    context.config.audio.diarization.model_id
                    or "pyannote/speaker-diarization-community-1"
                ).diarize(audio_path, video.video_id)
                align_speakers(transcript, diar)
            if (
                context.config.audio.audio_events.enabled
                and context.config.audio.audio_events.model_id
            ):
                events = AudioEventAdapter(
                    context.config.audio.audio_events.model_id, context.config.performance.device
                ).classify(audio_path, video.video_id)
                for event in events:
                    event.evidence_frame_ids = frames_for_interval(
                        frames, event.start_timestamp, event.end_timestamp
                    )
            return transcript, diar, events

        value = run_stage(context, video.video_id, "audio", key, compute, recoverable=True)
        if value is None:
            return [], [], []
        transcript, diar, events = value
        save_stage_cache(
            path,
            key,
            {
                "transcript": [x.model_dump(mode="json") for x in transcript],
                "diarization": [x.model_dump(mode="json") for x in diar],
                "events": [x.model_dump(mode="json") for x in events],
            },
        )
        return transcript, diar, events

    def _analyze(
        self,
        context: PipelineContext,
        video: VideoRecord,
        frames: list[FrameRecord],
        scenes: list[SceneRecord],
        detections: list[DetectionRecord],
        tracks: list[TrackRecord],
        observations: list[TrackObservation],
        visual: list[VisualResponse],
        transcript: list[TranscriptSegment],
        ocr: list[OCRRecord],
        key: str,
    ) -> AnalysisArtifacts:
        path = context.layout.cache / "analysis.json"
        cached = load_stage_cache(path, key)
        if cached:
            return (
                [AttributeRecord.model_validate(x) for x in cached["attributes"]],
                [RelationRecord.model_validate(x) for x in cached["relations"]],
                [EventRecord.model_validate(x) for x in cached["events"]],
                [NarrativeSegment.model_validate(x) for x in cached["narratives"]],
                [CodebookScore.model_validate(x) for x in cached["scores"]],
                [ModelAgreementRecord.model_validate(x) for x in cached["agreements"]],
            )

        def compute() -> AnalysisArtifacts:
            attributes: list[AttributeRecord] = []
            for frame in frames:
                frame_dets = [d for d in detections if d.frame_id == frame.frame_id]
                for name, value in frame_composition(frame_dets).items():
                    attributes.append(
                        AttributeRecord(
                            attribute_id=f"attr_{uuid.uuid4().hex[:16]}",
                            video_id=video.video_id,
                            frame_id=frame.frame_id,
                            name=name,
                            value=value,
                            confidence=1.0,
                            model_id="deterministic",
                        )
                    )
            for detection in detections:
                salience = compute_salience(detection)
                for name, value in salience.__dict__.items():
                    attributes.append(
                        AttributeRecord(
                            attribute_id=f"attr_{uuid.uuid4().hex[:16]}",
                            video_id=video.video_id,
                            frame_id=detection.frame_id,
                            detection_id=detection.detection_id,
                            name=f"salience_{name}",
                            value=value,
                            confidence=1.0,
                            model_id="deterministic",
                        )
                    )
                attributes.append(
                    AttributeRecord(
                        attribute_id=f"attr_{uuid.uuid4().hex[:16]}",
                        video_id=video.video_id,
                        frame_id=detection.frame_id,
                        detection_id=detection.detection_id,
                        name="salience_composite",
                        value=salience.composite,
                        confidence=1.0,
                        model_id="deterministic",
                    )
                )
                if detection.crop_path:
                    for name, value in crop_attributes(detection.crop_path).items():
                        attributes.append(
                            AttributeRecord(
                                attribute_id=f"attr_{uuid.uuid4().hex[:16]}",
                                video_id=video.video_id,
                                frame_id=detection.frame_id,
                                detection_id=detection.detection_id,
                                name=name,
                                value=value,
                                confidence=1.0,
                                model_id="deterministic",
                            )
                        )
            relations = infer_proximity_relations(observations, video.video_id)
            events = track_lifecycle_events(tracks)
            narratives = scenes_to_narrative(video.video_id, scenes)
            agreements = aggregate_model_responses(video.video_id, visual)
            scores: list[CodebookScore] = []
            if (
                context.config.research.codebook_path
                and context.config.research.codebook_path.exists()
            ):
                codebook = Codebook.from_yaml(context.config.research.codebook_path)
                for response in visual:
                    if response.pass_type != "contextual":
                        continue
                    for entry in codebook.entries:
                        parsed = response.parsed or {}
                        value = parsed.get(entry.variable_id)
                        if value is not None:
                            scores.append(
                                CodebookScore(
                                    score_id=f"score_{uuid.uuid4().hex[:16]}",
                                    video_id=video.video_id,
                                    unit_type=response.target_type,
                                    unit_id=response.target_id,
                                    variable_id=entry.variable_id,
                                    score=value,
                                    confidence=response.confidence,
                                    model_id=response.provenance.model_id,
                                    prompt_version=response.prompt_version,
                                )
                            )
            return attributes, relations, events, narratives, scores, agreements

        value = run_stage(context, video.video_id, "analyze", key, compute)
        assert value is not None
        attributes, relations, events, narratives, scores, agreements = value
        save_stage_cache(
            path,
            key,
            {
                "attributes": [x.model_dump(mode="json") for x in attributes],
                "relations": [x.model_dump(mode="json") for x in relations],
                "events": [x.model_dump(mode="json") for x in events],
                "narratives": [x.model_dump(mode="json") for x in narratives],
                "scores": [x.model_dump(mode="json") for x in scores],
                "agreements": [x.model_dump(mode="json") for x in agreements],
            },
        )
        return value

    def _provenance(self, run_id: str) -> RunProvenance:
        env = inspect_environment()
        device = detect_device(self.config.performance.device)
        return RunProvenance(
            run_id=run_id,
            package_version=__version__,
            git_commit=env.get("git_commit"),
            git_dirty=env.get("git_dirty"),
            python_version=sys.version,
            operating_system=str(env.get("operating_system")),
            cpu=str(env.get("cpu")),
            ram_bytes=env.get("ram_bytes"),
            gpu=env.get("gpu_name"),
            cuda_available=bool(env.get("cuda_available")),
            torch_version=env.get("torch_version"),
            transformers_version=env.get("dependencies", {}).get("transformers"),
            dependency_versions=env.get("dependencies", {}),
            configuration_hash=self.config.config_hash(),
            random_seed=self.config.project.random_seed,
            device=device.device,
            dtype=self.config.performance.dtype,
            extra={"public_result_schema": PUBLIC_RESULT_SCHEMA_VERSION},
        )
