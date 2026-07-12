# Visual models

legopolitics separates specialized visual models from general vision-language models.

- **Florence-2**: captioning, region descriptions, grounding, detection-style tasks, and OCR.
- **BLIP / BLIP-2**: captioning and visual question answering.
- **CLIP**: image-text similarity, zero-shot labels, and agreement analysis.
- **DINOv2**: visual embeddings, clustering, representative crops, and re-identification support.
- **Hugging Face image-text-to-text VLMs**: open-ended frame, crop, and sequence interpretation.

Multiple Hugging Face models can be configured under `vision.models.huggingface_vlms`. Their raw outputs and provenance remain separate so the ensemble layer can compare model agreement rather than overwriting predictions. See [Hugging Face VLM Ensemble](huggingface_vlm_ensemble.md).
