# Model License Register

The MIT License in the repository root applies to original package code and
project-authored documentation. It does not automatically license external or
user-supplied model weights.

Model identifiers in example configurations are references only. Users must review
the model card, repository license, acceptable-use conditions, access restrictions,
and downstream obligations before use.

| Capability | Example family | Bundled? | Review required? |
|---|---|---:|---:|
| Closed-set detection | User-supplied YOLO-compatible weights | Source repository only; excluded from wheel | Yes |
| Open-vocabulary detection | Grounding DINO | No | Yes |
| Captioning/OCR | Florence-2 | No | Yes |
| Captioning/VQA | BLIP / BLIP-2 | No | Yes |
| General VLM | SmolVLM 500M; SmolVLM2 500M/2.2B; Qwen2.5-VL 3B; Idefics3 8B; LLaVA-OneVision 7B | No | Yes |
| Similarity | CLIP | No | Yes |
| Embeddings | DINOv2 | No | Yes |
| Segmentation | SAM 2-compatible models | No | Yes |
| ASR | Whisper / Faster-Whisper | No | Yes |
| Diarization | pyannote models | No | Yes |


Exact model IDs referenced by the supplied GPU profiles:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
HuggingFaceTB/SmolVLM2-2.2B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
HuggingFaceM4/Idefics3-8B-Llama3
llava-hf/llava-onevision-qwen2-7b-ov-hf
```

The package stores identifiers only and does not redistribute these weights.
