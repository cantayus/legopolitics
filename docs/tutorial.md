# Complete Tutorial

This tutorial moves from a lightweight local run to a comprehensive multimodal research workflow. It uses a hypothetical input called `videos/example.mp4`; replace that path with your own video.

## 1. Create an environment

```bash
python -m venv .venv
```

Activate it:

=== "Windows PowerShell"

    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "macOS/Linux"

    ```bash
    source .venv/bin/activate
    ```

Install the base package from PyPI after release:

```bash
python -m pip install --upgrade pip
python -m pip install legopolitics
```

For development from a cloned repository:

```bash
python -m pip install -e ".[dev]"
```

Install FFmpeg separately and confirm it is on `PATH`.

## 2. Inspect the environment

```bash
legopolitics doctor
```

Review the output before enabling GPU models. The command reports whether CUDA, Transformers, Ultralytics, Whisper, pyannote, OCR adapters, FFmpeg, and the configured output directory are available.

## 3. Run the lightweight baseline

Use the included minimal configuration:

```bash
legopolitics analyze-video \
  --video "videos/example.mp4" \
  --config "configs/minimal.yaml" \
  --output "outputs/example_baseline"
```

This baseline provides:

- Video metadata and fingerprints
- Scene-aware hybrid sampling
- Sampled frame images
- Frame-quality measurements
- Composition and salience components
- SQLite run manifest
- Excel workbook
- JSON and JSONL records
- HTML report
- Parquet tables when `pyarrow` is installed

Validate the completed output:

```bash
legopolitics validate --output "outputs/example_baseline"
```

## 4. Compare sampling modes

Create copies of the minimal configuration and change:

```yaml
sampling:
  mode: "fixed"       # fixed, scene, hybrid, adaptive, or all
  fps: 1.0
```

The modes serve different research needs:

| Mode | Best use |
|---|---|
| `fixed` | Consistent longitudinal measurement and easy comparison across videos |
| `scene` | Compact scene summaries and videos with long static shots |
| `hybrid` | General research use; fixed coverage plus scene boundaries |
| `adaptive` | Rapid action, abrupt object changes, flashes, and high-motion sequences |
| `all` | Short videos or frame-level event reconstruction with sufficient compute |

The `Frames` sheet records why every frame was selected, allowing the sampling process to be audited.

## 5. Add a custom YOLO-compatible detector

Install the optional adapter:

```bash
python -m pip install "legopolitics[yolo]"
```

Review the applicable Ultralytics license before enabling the adapter. Then configure your weights:

```yaml
legal:
  ultralytics_license_acknowledged: true

detection:
  enabled: true
  primary_backend: "yolo"
  yolo:
    weights: "models/best.pt"
    task: "auto"
    confidence_threshold: 0.30
    iou_threshold: 0.50
    image_size: 1280
    device: "auto"
    half_precision: true
    save_annotated_frames: true
```

Run the analysis again:

```bash
legopolitics analyze-video \
  --video "videos/example.mp4" \
  --config "analysis_yolo.yaml" \
  --output "outputs/example_yolo"
```

Review:

- `Detections` sheet for class, confidence, coordinates, area, center, and model provenance
- `annotated_frames/` for visual checks
- `crops/` for figure and object crops
- `Errors` sheet for corrupt or failed items

## 6. Add open-vocabulary detection

Install:

```bash
python -m pip install "legopolitics[grounding]"
```

Configure research-specific text queries:

```yaml
detection:
  open_vocabulary:
    enabled: true
    backend: "grounding_dino"
    box_threshold: 0.30
    text_threshold: 0.25
    text_queries:
      - "brick-built figure"
      - "figure holding a weapon"
      - "national flag"
      - "missile"
      - "destroyed building"
      - "hospital"
      - "civilian vehicle"
```

Closed-set and open-vocabulary outputs remain separate. Fused detections are derived without deleting the original predictions.

## 7. Track recurring figures and objects

Enable tracking:

```yaml
tracking:
  enabled: true
  backend: "iou"
  tracker: "bytetrack"
  maximum_track_gap_frames: 10
  minimum_track_length_frames: 2
  iou_match_threshold: 0.30
  use_appearance_embeddings: true
  appearance_model: "dinov2"
  secondary_appearance_model: "clip"
```

Install the relevant visual dependencies when using DINOv2 or CLIP:

```bash
python -m pip install "legopolitics[huggingface,tracking]"
```

The output distinguishes detections from persistent tracks. Use the `Tracks` and `Track_Observations` sheets to study first appearance, last appearance, screen time, size, centrality, group assignment, role, and suspected identity switches.

## 8. Add segmentation

Install:

```bash
python -m pip install "legopolitics[segmentation]"
```

Configure:

```yaml
segmentation:
  enabled: true
  backend: "sam2"
  use_detection_boxes_as_prompts: true
  propagate_masks: true
  save_masks: true
  save_rgba_crops: true
```

Model availability and exact identifiers vary. Supply a local model path or supported model identifier and validate the adapter with a small video before running the full corpus.

## 9. Add visual-language models

Install:

```bash
python -m pip install "legopolitics[huggingface]"
```

Example configuration:

```yaml
vision:
  whole_frame: true
  crops: true
  context_crops: true
  track_representatives: true
  track_contact_sheets: true
  sequence_windows: true
  sequence_window_frames: 5
  models:
    florence2:
      enabled: true
      model_id: "microsoft/Florence-2-large-ft"
      tasks:
        - "caption"
        - "more_detailed_caption"
        - "object_detection"
        - "dense_region_caption"
        - "ocr"
        - "ocr_with_region"
    clip:
      enabled: true
      model_id: "openai/clip-vit-large-patch14"
      labels_from_codebook: true
    dinov2:
      enabled: true
      model_id: "facebook/dinov2-large"
```

The package uses model adapters rather than collapsing every output into one caption. Raw outputs, parsed outputs, prompts, model revisions, and ensemble results remain separately auditable.

## 10. Add OCR and text analysis

Install:

```bash
python -m pip install "legopolitics[ocr]"
```

Configure:

```yaml
ocr:
  enabled: true
  backends:
    - "florence2"
    - "trocr"
  detect_language: true
  translate_to: "en"
  temporal_deduplication: true
  associate_text_with_tracks: true
```

Temporal deduplication prevents a subtitle displayed across several frames from being counted as several independent statements.

## 11. Add speech and audio analysis

Install:

```bash
python -m pip install "legopolitics[audio,diarization]"
```

Configure:

```yaml
audio:
  enabled: true
  extract_audio: true
  asr:
    backend: "whisper"
    model: "openai/whisper-large-v3"
    task: "transcribe"
    word_timestamps: true
  diarization:
    enabled: true
    backend: "pyannote"
    model_id: "pyannote/speaker-diarization-community-1"
  translation:
    enabled: true
    target_language: "en"
  audio_events:
    enabled: true
    labels:
      - "explosion"
      - "gunshot"
      - "siren"
      - "cheering"
      - "crying"
      - "chanting"
      - "military-style music"
      - "silence"
```

Some diarization models require a Hugging Face token and acceptance of model terms. Store tokens in environment variables; do not put them in YAML or commit them to Git.

## 12. Supply event context and a research question

Context can improve interpretation, but it can also bias a model. The package therefore supports separate passes:

```yaml
research:
  event_context: >
    Provide carefully sourced factual background about the event,
    producer, date, and relevant actors.
  research_question: >
    How are the depicted groups presented in terms of agency,
    aggression, victimization, heroism, competence, legitimacy,
    and moral responsibility?
  codebook_path: "configs/propaganda_codebook.yaml"
  inference_constraints:
    - "Do not infer nationality solely from clothing color."
    - "Do not infer religion from appearance."
    - "Separate observation from interpretation."
    - "Do not treat the supplied event context as visual evidence."

prompting:
  blind_visual_pass: true
  contextual_pass: true
  codebook_pass: true
  contradiction_pass: true
  relationship_pass: true
  narrative_pass: true
  save_rendered_prompts: true
  save_raw_responses: true
```

The four principal interpretation stages are:

1. **Blind observation** — no historical context.
2. **Contextual interpretation** — context is available but does not count as direct evidence.
3. **Codebook coding** — predefined variables and scales.
4. **Contradiction pass** — evidence that weakens or complicates the interpretation.

## 13. Customize the political codebook

Copy `configs/propaganda_codebook.yaml` and edit the definitions for your study. Each variable can specify:

- Unit of analysis
- Scale and allowed values
- Inclusion and exclusion criteria
- Positive, negative, and ambiguous examples
- Required evidence
- Prompt instructions
- Aggregation method

Keep the codebook under version control. Its hash is stored in provenance so that results can be connected to the exact coding rules used.

## 14. Inspect relationships and narrative structure

The package can create frame- and scene-level relations such as:

- `aiming_weapon_at`
- `protecting`
- `restraining`
- `fleeing_from`
- `positioned_under_same_flag`
- `speaking_to`

It can also aggregate sequences into candidate narrative segments such as introduction, threat, victimization, attack, counterattack, rescue, victory, mourning, and call to action. These are model-supported interpretations rather than ground truth and should be validated.

## 15. Run several batches in parallel

For four workers, launch the following command four times and change `--batch-id` to `0`, `1`, `2`, and `3`:

```bash
legopolitics analyze-directory \
  --input-root "videos" \
  --output-root "outputs" \
  --config "configs/comprehensive.yaml" \
  --recursive \
  --num-batches 4 \
  --batch-id 0 \
  --resume
```

The video-to-batch assignment is deterministic. SQLite locks and atomic writes reduce duplicate work and partial artifacts.

## 16. Review validation samples

Install the local review interface:

```bash
python -m pip install "legopolitics[review]"
```

Launch it with a validation file:

```bash
legopolitics review \
  validation.jsonl \
  --corrections-file corrections.jsonl
```

Corrections are append-only. The package preserves automated predictions and constructs resolved views rather than erasing the original output.

## 17. Preserve reproducibility

For every published analysis, archive:

- Package version and Git commit
- Effective YAML configuration
- Custom YOLO weight hash
- Model IDs and revisions
- Prompt versions
- Political codebook
- Sampling settings
- Random seed
- Environment and dependency versions
- Human corrections and adjudication records

The package writes most of this information automatically to `provenance.json`, `environment.json`, `effective_config.yaml`, the SQLite manifest, and the Excel metadata sheets.

## 18. Recommended research workflow

1. Run the lightweight pipeline over the corpus.
2. Validate sampling and video integrity.
3. Validate YOLO detections on a stratified sample.
4. Enable tracking and inspect identity switches.
5. Add OCR and audio.
6. Add visual-language models and preserve separate outputs.
7. Compare blind and contextual interpretations.
8. Validate political codebook scores with human coders.
9. Estimate model and intercoder agreement.
10. Freeze the configuration, codebook, prompts, and model revisions before final analysis.
