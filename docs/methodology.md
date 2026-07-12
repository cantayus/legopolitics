# Methodology

The pipeline distinguishes six analysis passes:

1. Blind visual observation without political context.
2. Context-conditioned interpretation.
3. Research-codebook coding.
4. Contradiction and falsification review.
5. Subject–predicate–object relationship extraction.
6. Temporal narrative aggregation.

Deterministic measurements such as bounding-box area, centrality, frame quality, composition, color proportions, and track duration remain separate from generative model interpretations. Each record can reference frames, detections, tracks, OCR regions, and audio segments.

Researchers should validate detector classes, tracking continuity, OCR, transcripts, relations, and codebook scores against a human-coded sample. The package supplies sampling and agreement utilities but does not make automated outputs ground truth.
